"""Direct grouped-query attention for one-to-four offset-causal queries."""

import torch
import triton
import triton.language as tl

from shape_policy import gqa_block_size


@triton.jit
def _gqa_kernel(
    q_ptr, k_ptr, v_ptr, positions_ptr, out_ptr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr,
    stride_qq: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr,
    stride_ks: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vh: tl.constexpr,
    stride_vs: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr,
    stride_oq: tl.constexpr, stride_od: tl.constexpr,
    SCALE: tl.constexpr, CAPACITY: tl.constexpr, Q_LEN: tl.constexpr,
    GROUPS: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    group = rows // Q_LEN
    query_index = rows - group * Q_LEN
    query_head = kv_head * GROUPS + group
    row_mask = rows < GROUPS * Q_LEN
    q_offsets = (
        batch * stride_qb + query_head[:, None] * stride_qh
        + query_index[:, None] * stride_qq + dims[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_offsets, mask=row_mask[:, None], other=0.0)
    max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    log2e = 1.4426950408889634

    for block_start in range(0, CAPACITY, BLOCK_N):
        keys = block_start + tl.arange(0, BLOCK_N)
        k_offsets = (
            batch * stride_kb + kv_head * stride_kh
            + keys[None, :] * stride_ks + dims[:, None] * stride_kd
        )
        k = tl.load(k_ptr + k_offsets, mask=keys[None, :] < CAPACITY, other=0.0)
        scores = tl.dot(q, k) * SCALE
        query_positions = tl.load(
            positions_ptr + query_index, mask=row_mask, other=-1
        )
        visible = keys[None, :] <= query_positions[:, None]
        scores = tl.where(row_mask[:, None] & visible, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(max_score, block_max)
        alpha = tl.exp2((max_score - next_max) * log2e)
        probabilities = tl.exp2((scores - next_max[:, None]) * log2e)
        probabilities = tl.where(visible & row_mask[:, None], probabilities, 0.0)
        v_offsets = (
            batch * stride_vb + kv_head * stride_vh
            + keys[:, None] * stride_vs + dims[None, :] * stride_vd
        )
        values = tl.load(
            v_ptr + v_offsets, mask=keys[:, None] < CAPACITY, other=0.0
        )
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), values
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_score = next_max

    result = accumulator / normalizer[:, None]
    out_offsets = (
        batch * stride_ob + query_head[:, None] * stride_oh
        + query_index[:, None] * stride_oq + dims[None, :] * stride_od
    )
    tl.store(out_ptr + out_offsets, result, mask=row_mask[:, None])


def grouped_query_attention(query, key_cache, value_cache, positions, scale):
    """Apply direct GQA with absolute-position offset causality."""
    batch, query_heads, query_length, head_dim = query.shape
    _, kv_heads, capacity, cache_head_dim = key_cache.shape
    if not (query_length in (1, 2, 4) and head_dim == cache_head_dim == 128):
        raise ValueError("GQA expects Q in {1,2,4} and head width 128")
    groups = query_heads // kv_heads
    output = torch.empty_like(query)
    block_n = gqa_block_size(capacity, query_length)
    _gqa_kernel[(batch, kv_heads)](
        query, key_cache, value_cache, positions, output,
        *query.stride(), *key_cache.stride(), *value_cache.stride(),
        *output.stride(), SCALE=scale, CAPACITY=capacity,
        Q_LEN=query_length, GROUPS=groups, HEAD_DIM=head_dim,
        BLOCK_N=block_n, BLOCK_M=16,
        num_warps=4, num_stages=2,
    )
    return output
