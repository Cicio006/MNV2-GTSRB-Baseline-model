"""Public API smoke tests for Person A custom-kernel imports."""

from __future__ import annotations

from torch import nn

from src.backends.triton.kernels.ops import (
    custom_fused_linear_relu,
    custom_linear,
    custom_normalize,
    custom_relu,
    custom_relu6,
)
from src.backends.triton.kernels.wrappers import (
    CustomFusedLinearReLU,
    CustomLinear,
    CustomNormalize,
    CustomReLU,
    CustomReLU6,
)


def test_public_kernel_api_is_importable() -> None:
    assert callable(custom_relu)
    assert callable(custom_relu6)
    assert callable(custom_normalize)
    assert callable(custom_linear)
    assert callable(custom_fused_linear_relu)


def test_public_kernel_wrappers_are_modules() -> None:
    assert issubclass(CustomReLU, nn.Module)
    assert issubclass(CustomReLU6, nn.Module)
    assert issubclass(CustomNormalize, nn.Module)
    assert issubclass(CustomLinear, nn.Module)
    assert issubclass(CustomFusedLinearReLU, nn.Module)
