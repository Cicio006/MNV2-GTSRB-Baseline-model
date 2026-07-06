"""Public API for Person A's custom CUDA/Triton operations.

Callers should import this module instead of invoking raw Triton kernels.
"""

from __future__ import annotations

import torch

from src.backends.triton.kernels.fused_linear_relu_triton import fused_linear_relu_forward_triton
from src.backends.triton.kernels.linear_triton import linear_forward_triton
from src.backends.triton.kernels.norm_triton import normalize_forward_triton
from src.backends.triton.kernels.relu6_triton import relu6_forward_triton
from src.backends.triton.kernels.relu_triton import relu_forward_triton


def custom_relu(x: torch.Tensor) -> torch.Tensor:
    """Compute ReLU with the Triton ReLU forward kernel."""
    return relu_forward_triton(x)


def custom_relu6(x: torch.Tensor) -> torch.Tensor:
    """Compute ReLU6 with the Triton ReLU6 forward kernel."""
    return relu6_forward_triton(x)


def custom_normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Compute per-channel NCHW normalization with the Triton normalization kernel."""
    return normalize_forward_triton(x, mean, std)


def custom_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``torch.nn.functional.linear`` with the Triton linear kernel."""
    return linear_forward_triton(x, weight, bias)


def custom_fused_linear_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute fused Linear + Bias + ReLU with the Triton fused kernel."""
    return fused_linear_relu_forward_triton(x, weight, bias)
