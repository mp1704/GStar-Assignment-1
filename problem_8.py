# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def _flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal, GQA-aware FlashAttention-2 forward pass that additionally
    writes out the logsumexp `M` per query row. `M` lets the backward pass
    recompute the exact softmax probabilities for a (query, key) tile without
    having to redo the running-max bookkeeping, and without ever having
    materialized the full score matrix here either.
    """
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    d_offsets = tl.arange(0, HEAD_DIM)
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + d_offsets[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Every key up to the end of this query block can matter; the causal
    # mask below takes care of excluding the ones that don't.
    q_block_end = (q_block_idx + 1) * BLOCK_M
    for start_n in range(0, q_block_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        s_ij = tl.where(causal, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        acc *= alpha[:, None]
        l_i *= alpha

        p_ij = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij.to(v_block.dtype), v_block)

        m_i = m_new

    l_i_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    logsumexp = m_i + tl.log(l_i_safe)

    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + d_offsets[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, logsumexp, mask=q_offsets < SEQ_LEN)


@triton.jit
def _flash_attention_backward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, M_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per query block, same grid as the forward kernel. For
    every key/value block it attends to, it recomputes the attention
    probabilities (via the saved logsumexp `M`, so no running-max tracking
    is needed this time), then derives dQ, dK and dV from them.

    dQ is only ever touched by the single program that owns this query
    block, so it is accumulated in registers and stored once at the end.
    dK and dV, on the other hand, are shared: every query block whose
    window reaches a given key contributes to that key's gradient, and in
    GQA several query heads fall into the same K/V head too. Both are
    written with `tl.atomic_add` so concurrent programs don't clobber each
    other's contributions.
    """
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    d_offsets = tl.arange(0, HEAD_DIM)
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets[:, None] < SEQ_LEN

    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + d_offsets[None, :])
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    do_ptrs = dO_ptr + batch_idx * do_stride_b + q_head_idx * do_stride_h + \
               (q_offsets[:, None] * do_stride_s + d_offsets[None, :])
    do_block = tl.load(do_ptrs, mask=q_mask, other=0.0)

    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + d_offsets[None, :])
    o_block = tl.load(o_ptrs, mask=q_mask, other=0.0)

    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    m_block = tl.load(m_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)

    # D_i = rowsum(dO_i * O_i), the correction term in the softmax backward
    # that comes from how every probability in a row depends on every score.
    delta = tl.sum(do_block.to(tl.float32) * o_block.to(tl.float32), axis=1)

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_block_end = (q_block_idx + 1) * BLOCK_M
    for start_n in range(0, q_block_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets[:, None] < SEQ_LEN

        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Recompute P_ij directly from the saved logsumexp - no max-tracking
        # needed since M already is the full logsumexp of the row.
        s_ij = tl.dot(q_block, k_block) * softmax_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        s_ij = tl.where(causal, s_ij, -float('inf'))
        p_ij = tl.exp(s_ij - m_block[:, None])

        dv_contrib = tl.dot(tl.trans(p_ij).to(do_block.dtype), do_block)
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
                   (k_offsets[:, None] * dv_stride_s + d_offsets[None, :])
        tl.atomic_add(dv_ptrs, dv_contrib, mask=k_mask)

        dp_ij = tl.dot(do_block, tl.trans(v_block))
        ds_ij = p_ij * (dp_ij - delta[:, None])
        ds_ij = ds_ij * softmax_scale

        dq_acc += tl.dot(ds_ij.to(k_block.dtype), tl.trans(k_block))

        dk_contrib = tl.dot(tl.trans(ds_ij).to(q_block.dtype), q_block)
        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
                   (k_offsets[:, None] * dk_stride_s + d_offsets[None, :])
        tl.atomic_add(dk_ptrs, dk_contrib, mask=k_mask)

    dq_ptrs = dQ_ptr + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + \
               (q_offsets[:, None] * dq_stride_s + d_offsets[None, :])
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=q_mask)


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

        _flash_attention_forward_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads
        softmax_scale = ctx.softmax_scale

        dq = torch.empty_like(q)
        # dK/dV are written from many programs at once via atomic_add, so we
        # accumulate them in float32 for numerical safety and cast down to
        # the parameter dtype once we're done.
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        do = do.contiguous()

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

        _flash_attention_backward_kernel[grid](
            q, k, v, o, do, M,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)
