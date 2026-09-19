"""Fused vocabulary argmax and speculative-token acceptance."""

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_accept_kernel(
    logits_ptr, input_ids_ptr, targets_ptr, matches_ptr,
    stride_lb: tl.constexpr, stride_lq: tl.constexpr,
    stride_ib: tl.constexpr, stride_iq: tl.constexpr,
    VOCAB: tl.constexpr, Q_LEN: tl.constexpr, BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    query = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    best_value = -float("inf")
    best_index = VOCAB
    for start in range(0, VOCAB, BLOCK):
        indices = start + offsets
        values = tl.load(
            logits_ptr + batch * stride_lb + query * stride_lq + indices,
            mask=indices < VOCAB, other=-float("inf"),
        ).to(tl.float32)
        block_value = tl.max(values, axis=0)
        block_index = tl.min(tl.where(values == block_value, indices, VOCAB), axis=0)
        take = (block_value > best_value) | (
            (block_value == best_value) & (block_index < best_index)
        )
        best_value = tl.where(take, block_value, best_value)
        best_index = tl.where(take, block_index, best_index)
    tl.store(targets_ptr + batch * Q_LEN + query, best_index)
    is_proposal = query + 1 < Q_LEN
    proposed = tl.load(
        input_ids_ptr + batch * stride_ib + (query + 1) * stride_iq,
        mask=is_proposal,
        other=-1,
    )
    tl.store(
        matches_ptr + batch * (Q_LEN - 1) + query,
        best_index == proposed,
        mask=is_proposal,
    )


def argmax_and_accept(logits: torch.Tensor, input_ids: torch.Tensor):
    """Return target argmax IDs and following-proposal match flags."""
    batch, query_length, vocab = logits.shape
    targets = torch.empty((batch, query_length), dtype=torch.int32, device=logits.device)
    matches = torch.empty(
        (batch, max(query_length - 1, 1)), dtype=torch.uint8, device=logits.device
    )
    _argmax_accept_kernel[(batch, query_length)](
        logits, input_ids, targets, matches,
        logits.stride(0), logits.stride(1), input_ids.stride(0), input_ids.stride(1),
        VOCAB=vocab, Q_LEN=query_length, BLOCK=1024, num_warps=8,
    )
    return targets, matches
