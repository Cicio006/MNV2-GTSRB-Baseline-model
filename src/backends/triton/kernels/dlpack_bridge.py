"""DLPack handoff helpers for CuPy/PyTorch/Triton integration experiments.

These helpers are intentionally small and optional. Person A's kernels consume
``torch.Tensor`` values; if Person B keeps the reference path in CuPy, Person C
can use these helpers or centralize equivalent logic in an integration module.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils import dlpack


def cupy_to_torch(cupy_array: Any) -> torch.Tensor:
    """Convert a CuPy-like array with ``__dlpack__`` support to ``torch.Tensor``."""
    if not hasattr(cupy_array, "__dlpack__"):
        raise TypeError("cupy_array must provide __dlpack__ for zero-copy handoff")
    return dlpack.from_dlpack(cupy_array)


def torch_to_cupy_dlpack(torch_tensor: torch.Tensor) -> Any:
    """Return a DLPack capsule that CuPy can consume with ``cp.from_dlpack``."""
    if not isinstance(torch_tensor, torch.Tensor):
        raise TypeError("torch_tensor must be a torch.Tensor")
    return dlpack.to_dlpack(torch_tensor)
