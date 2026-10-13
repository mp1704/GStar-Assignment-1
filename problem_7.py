import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
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
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    
    qk_scale = softmax_scale * 1.44269504

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    q_block_start = q_block_idx * BLOCK_M

    # Block-aligned start of the sliding window, same idea as Problem 6.
    window_start = tl.maximum(0, q_block_start - WINDOW_SIZE + 1)
    window_start = (window_start // BLOCK_N) * BLOCK_N

    # The sink tokens live at the very start of the sequence. Whenever the
    # window already reaches back that far, those tokens are handled by the
    # window loop instead, so Phase 0 should stop wherever the window begins.
    sink_end = tl.minimum(SINK_SIZE, window_start)

    # --- Phase 0: Sink Blocks ---
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale

        # Sink tokens always precede this query block, so only the sink
        # boundary itself needs to be masked here.
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(is_sink, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        acc *= alpha[:, None]
        l_i *= alpha

        p_ij = tl.math.exp2(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij.to(v_block.dtype), v_block)

        m_i = m_new

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale

        # A key is valid if it's inside the sliding window OR it's a sink
        # token. The sink check matters near the start of the sequence,
        # where window_start can fall below SINK_SIZE for some rows in
        # this block even though the block-level window_start already
        # passed it.
        in_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(in_window | is_sink, s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        # A row can have no valid key at all in this tile (its true window
        # hasn't started yet, and no sink token lives in this tile either)
        # - then m_i and m_new are both -inf, and exp2(-inf - -inf) is NaN,
        # not 0. Guard it explicitly.
        no_valid_yet = m_new == -float('inf')
        alpha = tl.where(no_valid_yet, 0.0, tl.math.exp2(m_i - m_new))
        acc *= alpha[:, None]
        l_i *= alpha

        p_ij = tl.where(no_valid_yet[:, None], 0.0, tl.math.exp2(s_ij - m_new[:, None]))
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij.to(v_block.dtype), v_block)

        m_i = m_new

    # --- Phase 2: Diagonal Block ---
    for start_n in range(q_block_start, q_block_start + BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block, k_block) * qk_scale

        causal = q_offsets[:, None] >= k_offsets[None, :]
        in_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        is_sink = k_offsets[None, :] < SINK_SIZE
        s_ij = tl.where(causal & (in_window | is_sink), s_ij, -float('inf'))

        m_new = tl.maximum(m_i, tl.max(s_ij, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        acc *= alpha[:, None]
        l_i *= alpha

        p_ij = tl.math.exp2(s_ij - m_new[:, None])
        l_i += tl.sum(p_ij, axis=1)
        acc += tl.dot(p_ij.to(v_block.dtype), v_block)

        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
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
    return o