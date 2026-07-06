"""Correctness tests for the Triton ReLU6 kernel."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from src.backends.triton.kernels.ops import custom_relu6
from src.backends.triton.kernels.relu6_triton import relu6_forward_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel tests require a CUDA GPU"
)


@pytest.mark.parametrize(
    "shape",
    [
        (257,),
        (1, 43),
        (8, 43),
        (32, 43),
        (1, 1280),
        (8, 1280),
        (32, 1280),
        (1, 32, 64, 64),
        (8, 32, 64, 64),
        (1, 64, 16, 16),
        (8, 64, 16, 16),
        (1, 1280, 4, 4),
        (8, 1280, 4, 4),
    ],
)
def test_relu6_forward_triton_matches_torch_clamp(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, device="cuda", dtype=torch.float32) * 8.0 - 4.0

    expected = torch.clamp(x, min=0.0, max=6.0)
    actual = relu6_forward_triton(x)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_custom_relu6_makes_non_contiguous_input_contiguous() -> None:
    x = torch.randn((8, 64), device="cuda", dtype=torch.float32).transpose(0, 1)

    actual = custom_relu6(x)
    expected = torch.clamp(x, min=0.0, max=6.0)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
