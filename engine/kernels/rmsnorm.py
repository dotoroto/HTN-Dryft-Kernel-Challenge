"""Decode RMSNorm with the reference's BF16 cast placement."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr, weight_ptr, out_ptr, row_stride, width, epsilon, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < width
    offsets = row * row_stride + columns
    values = tl.load(x_ptr + offsets, mask=valid, other=0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / width
    normalized = values * tl.math.rsqrt(variance + epsilon)
    weight = tl.load(weight_ptr + columns, mask=valid, other=0)
    # Qwen3 casts the normalized value to BF16 before multiplying by weight.
    result = normalized.to(out_ptr.dtype.element_ty) * weight
    tl.store(out_ptr + offsets, result, mask=valid)


def rms_norm_decode(x: torch.Tensor, weight: torch.Tensor, epsilon: float):
    """Normalize contiguous [B,K,D] decode activations without model dispatch."""
    if not x.is_contiguous():
        x = x.contiguous()
    width = x.shape[-1]
    rows = x.numel() // width
    output = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    _rms_norm_kernel[(rows,)](
        x, weight, output, width, width, epsilon,
        BLOCK=block, num_warps=4,
    )
    return output
