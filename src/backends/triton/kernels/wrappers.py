"""``torch.nn.Module`` wrappers for custom kernel operations."""

from __future__ import annotations

import torch
from torch import nn

from src.backends.triton.kernels.ops import (
    custom_fused_linear_relu,
    custom_linear,
    custom_normalize,
    custom_relu,
    custom_relu6,
)


class CustomReLU(nn.Module):
    """Module wrapper around :func:`custom_relu`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_relu(x)


class CustomReLU6(nn.Module):
    """Module wrapper around :func:`custom_relu6`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_relu6(x)


class CustomNormalize(nn.Module):
    """Module wrapper around :func:`custom_normalize`."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.to(dtype=torch.float32))
        self.register_buffer("std", std.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_normalize(x, self.mean.to(device=x.device), self.std.to(device=x.device))


class CustomLinear(nn.Module):
    """Module wrapper around :func:`custom_linear`."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(dtype=torch.float32))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(device=x.device) if self.bias is not None else None
        return custom_linear(x, self.weight.to(device=x.device), bias)


class CustomFusedLinearReLU(nn.Module):
    """Module wrapper around :func:`custom_fused_linear_relu`."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(dtype=torch.float32))
        self.register_buffer("bias", bias.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_fused_linear_relu(
            x,
            self.weight.to(device=x.device),
            self.bias.to(device=x.device),
        )
