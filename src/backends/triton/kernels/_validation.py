"""Shared validation helpers for custom kernel wrappers."""

from __future__ import annotations

import torch


def ensure_cuda_float32_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Validate the public kernel tensor contract and return a contiguous tensor."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    return tensor.contiguous() if not tensor.is_contiguous() else tensor
