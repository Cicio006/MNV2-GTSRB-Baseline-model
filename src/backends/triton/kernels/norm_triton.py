"""Triton implementation of per-channel image normalization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.backends.triton.kernels._validation import ensure_cuda_float32_tensor


@triton.jit
def _normalize_forward_kernel(
    x_ptr,
    mean_ptr,
    std_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    spatial_size = height * width
    channel_offsets = (offsets // spatial_size) % channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_offsets, mask=mask, other=0.0)
    std = tl.load(std_ptr + channel_offsets, mask=mask, other=1.0)
    y = (x - mean) / std
    tl.store(y_ptr + offsets, y, mask=mask)


def normalize_forward_triton(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Normalize a contiguous NCHW CUDA tensor using per-channel mean and std."""
    x = ensure_cuda_float32_tensor(x, "x")
    mean = ensure_cuda_float32_tensor(mean, "mean")
    std = ensure_cuda_float32_tensor(std, "std")

    if x.ndim != 4:
        raise ValueError("x must have shape [B, C, H, W]")
    batch_size, channels, height, width = x.shape
    if mean.numel() != channels:
        raise ValueError(f"mean must contain {channels} values")
    if std.numel() != channels:
        raise ValueError(f"std must contain {channels} values")

    y = torch.empty_like(x)
    n_elements = batch_size * channels * height * width
    if n_elements == 0:
        return y

    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    _normalize_forward_kernel[grid](
        x,
        mean,
        std,
        y,
        n_elements,
        channels=channels,
        height=height,
        width=width,
        block_size=block_size,
    )
    return y
