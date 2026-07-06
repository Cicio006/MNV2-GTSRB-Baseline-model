"""Correctness tests for the Week 3 Triton ReLU kernel."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from src.backends.triton.kernels.ops import custom_relu
from src.backends.triton.kernels.relu_triton import relu_forward_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel tests require a CUDA GPU"
)


@pytest.mark.parametrize(
    "shape",
    [
        (257,),
        (1, 43),
        (8, 43),
        (16, 43),
        (32, 43),
        (64, 43),
        (128, 43),
        (1, 1280),
        (8, 1280),
        (16, 1280),
        (32, 1280),
        (64, 1280),
        (128, 1280),
        (1, 32, 64, 64),
        (8, 32, 64, 64),
        (16, 32, 64, 64),
        (32, 32, 64, 64),
        (1, 64, 16, 16),
        (8, 64, 16, 16),
        (16, 64, 16, 16),
        (32, 64, 16, 16),
        (1, 1280, 4, 4),
        (8, 1280, 4, 4),
        (16, 1280, 4, 4),
        (32, 1280, 4, 4),
    ],
)
def test_relu_forward_triton_matches_torch(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, device="cuda", dtype=torch.float32) * 4.0 - 2.0

    expected = torch.relu(x)
    actual = relu_forward_triton(x)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_custom_relu_makes_non_contiguous_input_contiguous() -> None:
    x = torch.randn((8, 64), device="cuda", dtype=torch.float32).transpose(0, 1)

    actual = custom_relu(x)
    expected = torch.relu(x)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
