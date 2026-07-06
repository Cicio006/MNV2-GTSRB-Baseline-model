"""Correctness tests for the Triton fused Linear + Bias + ReLU kernel API."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from src.backends.triton.kernels.fused_linear_relu_triton import fused_linear_relu_forward_triton
from src.backends.triton.kernels.ops import custom_fused_linear_relu

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel tests require a CUDA GPU"
)


@pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64, 128])
def test_fused_linear_relu_forward_triton_matches_torch(batch_size: int) -> None:
    x = torch.randn((batch_size, 1280), device="cuda", dtype=torch.float32)
    weight = torch.randn((43, 1280), device="cuda", dtype=torch.float32) * 0.01
    bias = torch.randn((43,), device="cuda", dtype=torch.float32)

    expected = torch.relu(F.linear(x, weight, bias))
    actual = fused_linear_relu_forward_triton(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_custom_fused_linear_relu_supports_leading_dimensions() -> None:
    x = torch.randn((2, 4, 1280), device="cuda", dtype=torch.float32)
    weight = torch.randn((43, 1280), device="cuda", dtype=torch.float32) * 0.01
    bias = torch.randn((43,), device="cuda", dtype=torch.float32)

    actual = custom_fused_linear_relu(x, weight, bias)
    expected = torch.relu(F.linear(x, weight, bias))

    assert actual.shape == (2, 4, 43)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
