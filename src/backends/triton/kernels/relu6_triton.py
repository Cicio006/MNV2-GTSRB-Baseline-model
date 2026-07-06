"""Triton implementation of ReLU6 forward for contiguous float32 CUDA tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._validation import ensure_cuda_float32_tensor


@triton.jit
def _relu6_forward_kernel(x_ptr, y_ptr, n_elements: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(y_ptr + offsets, y, mask=mask)


def relu6_forward_triton(x: torch.Tensor) -> torch.Tensor:
    """Return ``torch.clamp(x, min=0.0, max=6.0)`` computed by Triton."""
    x = ensure_cuda_float32_tensor(x, "x")
    y = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return y

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _relu6_forward_kernel[grid](x, y, n_elements, block_size=block_size)
    return y
