"""Triton fused Linear + Bias + ReLU forward kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from src.backends.triton.kernels._validation import ensure_cuda_float32_tensor


@triton.jit
def _fused_linear_relu_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_start in range(0, k_size, block_k):
        k_offsets = k_start + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * k_size + k_offsets[None, :],
            mask=(offs_m[:, None] < m_size) & (k_offsets[None, :] < k_size),
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + offs_n[None, :] * k_size + k_offsets[:, None],
            mask=(offs_n[None, :] < n_size) & (k_offsets[:, None] < k_size),
            other=0.0,
        )
        acc += tl.dot(x, weight)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
    acc = tl.maximum(acc + bias[None, :], 0.0)
    tl.store(
        y_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
    )


def fused_linear_relu_forward_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return ``torch.relu(torch.nn.functional.linear(x, weight, bias))``."""
    x = ensure_cuda_float32_tensor(x, "x")
    weight = ensure_cuda_float32_tensor(weight, "weight")
    bias = ensure_cuda_float32_tensor(bias, "bias")

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [out_features, in_features]")

    in_features = x.shape[-1]
    out_features, weight_in_features = weight.shape
    if in_features != weight_in_features:
        raise ValueError(
            f"x last dimension ({in_features}) must match weight input features ({weight_in_features})"
        )
    if bias.numel() != out_features:
        raise ValueError(f"bias must contain {out_features} values")

    output = torch.empty((*x.shape[:-1], out_features), device=x.device, dtype=x.dtype)
    m_size = x.numel() // in_features
    if m_size == 0 or out_features == 0:
        return output

    x_2d = x.reshape(m_size, in_features)
    y_2d = output.reshape(m_size, out_features)
    block_m = 16
    block_n = 16
    block_k = 32
    grid = (triton.cdiv(m_size, block_m), triton.cdiv(out_features, block_n))
    _fused_linear_relu_forward_kernel[grid](
        x_2d,
        weight,
        bias,
        y_2d,
        m_size,
        out_features,
        in_features,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    return output
