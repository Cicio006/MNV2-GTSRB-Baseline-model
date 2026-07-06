"""Correctness tests for the Triton Linear kernel API."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from src.backends.triton.kernels.linear_triton import linear_forward_triton
from src.backends.triton.kernels.ops import custom_linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel tests require a CUDA GPU"
)


@pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("in_features,out_features", [(1280, 43), (256, 43)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_linear_forward_triton_matches_torch(
    batch_size: int,
    in_features: int,
    out_features: int,
    use_bias: bool,
) -> None:
    x = torch.randn((batch_size, in_features), device="cuda", dtype=torch.float32)
    weight = torch.randn((out_features, in_features), device="cuda", dtype=torch.float32) * 0.01
    bias = torch.randn((out_features,), device="cuda", dtype=torch.float32) if use_bias else None

    expected = F.linear(x, weight, bias)
    actual = linear_forward_triton(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_custom_linear_supports_leading_dimensions() -> None:
    x = torch.randn((2, 4, 1280), device="cuda", dtype=torch.float32)
    weight = torch.randn((43, 1280), device="cuda", dtype=torch.float32) * 0.01
    bias = torch.randn((43,), device="cuda", dtype=torch.float32)

    actual = custom_linear(x, weight, bias)
    expected = F.linear(x, weight, bias)

    assert actual.shape == (2, 4, 43)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
