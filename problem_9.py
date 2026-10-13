import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """GQA + Sliding Window + Attention Sink forward pass (same algorithm as
    problem_7), extended to also write out the logsumexp `M` per row so the
    backward pass below can recompute probabilities without redoing the
    online-softmax bookkeeping.
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

    q_block_start = q_block_idx * BLOCK_M

    # Block-aligned window start, as in problem_6/7.
    window_start = tl.maximum(0, q_block_start - WINDOW_SIZE + 1)
    window_start = (window_start // BLOCK_N) * BLOCK_N

    # Sink tokens are handled by Phase 0 only up to where the window phase
    # takes over, so the two never double-count the same key.
    sink_end = tl.minimum(SINK_SIZE, window_start)

    # --- Phase 0: Sink Blocks ---
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(is_sink, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        acc *= alpha[:, None]
        l_i *= alpha
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij.to(v_block.dtype), v_block)
        m_i = m_new

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        in_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(in_window | is_sink, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.exp(m_i - m_new)
        acc *= alpha[:, None]
        l_i *= alpha
        p_ij = tl.exp(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij.to(v_block.dtype), v_block)
        m_i = m_new

    # --- Phase 2: Diagonal Block ---
    for start_n in range(q_block_start, q_block_start + BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        in_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(causal & (in_window | is_sink), s_ij, -float('inf'))

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

    o_ptrs = O_ptr + batch_idx * o_stride_b + q_head_idx * o_stride_h + \
             (q_offsets[:, None] * o_stride_s + d_offsets[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, logsumexp, mask=q_offsets < SEQ_LEN)


@triton.jit
def _bwd_preprocess_kernel(
    O_ptr, dO_ptr, D_ptr,
    o_stride_b, o_stride_h, o_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    d_stride_b, d_stride_h, d_stride_s,
    SEQ_LEN,
    N_Q_HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Computes D_i = rowsum(dO_i * O_i) for every query row, once, so the
    main backward kernel below doesn't have to reload O on every K/V tile.
    """
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    head_idx = batch_head_idx % N_Q_HEADS

    d_offsets = tl.arange(0, HEAD_DIM)
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_offsets[:, None] < SEQ_LEN

    o_ptrs = O_ptr + batch_idx * o_stride_b + head_idx * o_stride_h + \
             (q_offsets[:, None] * o_stride_s + d_offsets[None, :])
    o_block = tl.load(o_ptrs, mask=q_mask, other=0.0)

    do_ptrs = dO_ptr + batch_idx * do_stride_b + head_idx * do_stride_h + \
               (q_offsets[:, None] * do_stride_s + d_offsets[None, :])
    do_block = tl.load(do_ptrs, mask=q_mask, other=0.0)

    delta = tl.sum(o_block.to(tl.float32) * do_block.to(tl.float32), axis=1)

    d_ptrs = D_ptr + batch_idx * d_stride_b + head_idx * d_stride_h + q_offsets * d_stride_s
    tl.store(d_ptrs, delta, mask=q_offsets < SEQ_LEN)


@triton.jit
def _flash_attention_backward_swa_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, dO_ptr, M_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    d_stride_b, d_stride_h, d_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE: int,
    N_Q_HEADS: int,
    N_KV_HEADS: int,
    SEQ_LEN: int,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Tile Sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Backward pass for GQA + Sliding Window + Attention Sinks. One program
    per query block (same grid as the forward kernel), walking the exact
    same sink / window / diagonal phases so it only ever recomputes the
    (query, key) pairs that were actually attended to in the forward pass.

    dQ belongs exclusively to this program and is stored directly. dK and dV
    are shared across query blocks (and, under GQA, across every query head
    in a group), so they're updated with `tl.atomic_add`.
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

    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    m_block = tl.load(m_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)

    d_ptrs = D_ptr + batch_idx * d_stride_b + q_head_idx * d_stride_h + q_offsets * d_stride_s
    delta = tl.load(d_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_block_start = q_block_idx * BLOCK_M
    window_start = tl.maximum(0, q_block_start - WINDOW_SIZE + 1)
    window_start = (window_start // BLOCK_N) * BLOCK_N
    sink_end = tl.minimum(SINK_SIZE, window_start)

    # --- Phase 0: Sink Blocks ---
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets[:, None] < SEQ_LEN
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_mask, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(is_sink, s_ij, -float('inf'))
        p_ij = tl.exp(s_ij - m_block[:, None])

        dv_contrib = tl.dot(tl.trans(p_ij).to(do_block.dtype), do_block)
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
                   (k_offsets[:, None] * dv_stride_s + d_offsets[None, :])
        tl.atomic_add(dv_ptrs, dv_contrib, mask=k_mask)

        dp_ij = tl.dot(do_block, tl.trans(v_block))
        ds_ij = p_ij * (dp_ij - delta[:, None]) * softmax_scale
        dq_acc += tl.dot(ds_ij.to(k_block.dtype), tl.trans(k_block))

        dk_contrib = tl.dot(tl.trans(ds_ij).to(q_block.dtype), q_block)
        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
                   (k_offsets[:, None] * dk_stride_s + d_offsets[None, :])
        tl.atomic_add(dk_ptrs, dk_contrib, mask=k_mask)

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets[:, None] < SEQ_LEN
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_mask, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        in_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(in_window | is_sink, s_ij, -float('inf'))
        p_ij = tl.exp(s_ij - m_block[:, None])

        dv_contrib = tl.dot(tl.trans(p_ij).to(do_block.dtype), do_block)
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
                   (k_offsets[:, None] * dv_stride_s + d_offsets[None, :])
        tl.atomic_add(dv_ptrs, dv_contrib, mask=k_mask)

        dp_ij = tl.dot(do_block, tl.trans(v_block))
        ds_ij = p_ij * (dp_ij - delta[:, None]) * softmax_scale
        dq_acc += tl.dot(ds_ij.to(k_block.dtype), tl.trans(k_block))

        dk_contrib = tl.dot(tl.trans(ds_ij).to(q_block.dtype), q_block)
        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
                   (k_offsets[:, None] * dk_stride_s + d_offsets[None, :])
        tl.atomic_add(dk_ptrs, dk_contrib, mask=k_mask)

    # --- Phase 2: Diagonal Block ---
    for start_n in range(q_block_start, q_block_start + BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets[:, None] < SEQ_LEN
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + d_offsets[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + d_offsets[None, :])
        v_block = tl.load(v_ptrs, mask=k_mask, other=0.0)

        s_ij = tl.dot(q_block, k_block) * softmax_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        in_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(causal & (in_window | is_sink), s_ij, -float('inf'))
        p_ij = tl.exp(s_ij - m_block[:, None])

        dv_contrib = tl.dot(tl.trans(p_ij).to(do_block.dtype), do_block)
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
                   (k_offsets[:, None] * dv_stride_s + d_offsets[None, :])
        tl.atomic_add(dv_ptrs, dv_contrib, mask=k_mask)

        dp_ij = tl.dot(do_block, tl.trans(v_block))
        ds_ij = p_ij * (dp_ij - delta[:, None]) * softmax_scale
        dq_acc += tl.dot(ds_ij.to(k_block.dtype), tl.trans(k_block))

        dk_contrib = tl.dot(tl.trans(ds_ij).to(q_block.dtype), q_block)
        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
                   (k_offsets[:, None] * dk_stride_s + d_offsets[None, :])
        tl.atomic_add(dk_ptrs, dk_contrib, mask=k_mask)

    dq_ptrs = dQ_ptr + batch_idx * dq_stride_b + q_head_idx * dq_stride_h + \
               (q_offsets[:, None] * dq_stride_s + d_offsets[None, :])
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=q_mask)


class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)


        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q)
        # Written concurrently from many programs via atomic_add (and, under
        # GQA, by several query heads at once), so accumulate in float32 and
        # cast down once at the end.
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)

        do = do.contiguous()

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

        D = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)
        _bwd_preprocess_kernel[grid](
            o, do, D,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            seq_len,
            n_q_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
        )

        _flash_attention_backward_swa_kernel[grid](
            q, k, v, do, M, D,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            softmax_scale,
            batch,
            n_q_heads,
            n_kv_heads,
            seq_len,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None

def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)
