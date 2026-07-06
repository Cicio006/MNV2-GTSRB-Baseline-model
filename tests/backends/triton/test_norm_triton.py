"""Correctness tests for the Week 3 Triton normalization kernel."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from src.backends.triton.kernels.norm_triton import normalize_forward_triton
from src.backends.triton.kernels.ops import custom_normalize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel tests require a CUDA GPU"
)


@pytest.mark.parametrize("batch_size", [1, 8, 16, 32, 64, 128])
def test_normalize_forward_triton_matches_torch(batch_size: int) -> None:
    x = torch.rand((batch_size, 3, 64, 64), device="cuda", dtype=torch.float32)
    mean = torch.tensor([0.3403, 0.3121, 0.3214], device="cuda", dtype=torch.float32)
    std = torch.tensor([0.2724, 0.2608, 0.2669], device="cuda", dtype=torch.float32)

    expected = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    actual = normalize_forward_triton(x, mean, std)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_custom_normalize_makes_non_contiguous_input_contiguous() -> None:
    x = torch.rand((3, 2, 64, 64), device="cuda", dtype=torch.float32).transpose(0, 1)
    mean = torch.tensor([0.3403, 0.3121, 0.3214], device="cuda", dtype=torch.float32)
    std = torch.tensor([0.2724, 0.2608, 0.2669], device="cuda", dtype=torch.float32)

    expected = (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
    actual = custom_normalize(x, mean, std)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
