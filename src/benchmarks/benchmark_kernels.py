"""Synthetic benchmarks for Person A's Triton kernels.

Run on a CUDA node with:

    PYTHONPATH=. python -m src.benchmarks.benchmark_kernels

These benchmarks are M5 presentation scaffolding. Replace synthetic tensors with
Person B reference tensors before M6/M7 comparisons.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.backends.triton.kernels.mobilenet_contract import BATCH_SIZES
from src.backends.triton.kernels.ops import (
    custom_fused_linear_relu,
    custom_linear,
    custom_normalize,
    custom_relu,
    custom_relu6,
)

DEFAULT_OUTPUT_CSV = Path("results/tables/kernel_benchmarks.csv")


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    batch_size: int
    shape: str
    reference: str
    custom_mean_ms: float
    custom_median_ms: float
    custom_std_ms: float
    reference_mean_ms: float
    speedup_vs_reference: float
    max_abs_error: float
    mean_abs_error: float


def _time_cuda_samples(fn: Callable[[], torch.Tensor], *, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton kernel benchmarks")


def _summarize_case(
    *,
    name: str,
    batch_size: int,
    shape: str,
    custom_fn: Callable[[], torch.Tensor],
    reference_fn: Callable[[], torch.Tensor],
    warmup: int,
    repeat: int,
) -> BenchmarkResult:
    custom_output = custom_fn()
    reference_output = reference_fn()
    torch.cuda.synchronize()
    diff = (custom_output - reference_output).abs()

    custom_samples = _time_cuda_samples(custom_fn, warmup=warmup, repeat=repeat)
    reference_samples = _time_cuda_samples(reference_fn, warmup=warmup, repeat=repeat)
    custom_mean = statistics.fmean(custom_samples)
    reference_mean = statistics.fmean(reference_samples)

    return BenchmarkResult(
        name=name,
        batch_size=batch_size,
        shape=shape,
        reference="torch",
        custom_mean_ms=custom_mean,
        custom_median_ms=statistics.median(custom_samples),
        custom_std_ms=statistics.pstdev(custom_samples) if len(custom_samples) > 1 else 0.0,
        reference_mean_ms=reference_mean,
        speedup_vs_reference=reference_mean / custom_mean if custom_mean > 0 else float("inf"),
        max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
    )


def run_benchmarks(*, warmup: int = 10, repeat: int = 50) -> list[BenchmarkResult]:
    """Run synthetic MobileNet-shaped benchmarks for Person A kernels."""
    _require_cuda()
    device = torch.device("cuda")
    results: list[BenchmarkResult] = []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
    classifier_weight = torch.randn((43, 1280), device=device, dtype=torch.float32) * 0.01
    classifier_bias = torch.randn((43,), device=device, dtype=torch.float32)

    for batch_size in BATCH_SIZES:
        image = torch.rand((batch_size, 3, 64, 64), device=device, dtype=torch.float32)
        logits_like = torch.randn((batch_size, 43), device=device, dtype=torch.float32)
        flattened = torch.randn((batch_size, 1280), device=device, dtype=torch.float32)
        feature_map = torch.randn((batch_size, 1280, 4, 4), device=device, dtype=torch.float32)

        cases: tuple[tuple[str, str, Callable[[], torch.Tensor], Callable[[], torch.Tensor]], ...] = (
            (
                "normalize",
                str(tuple(image.shape)),
                lambda: custom_normalize(image, mean, std),
                lambda: (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1),
            ),
            (
                "relu_logits",
                str(tuple(logits_like.shape)),
                lambda: custom_relu(logits_like),
                lambda: torch.relu(logits_like),
            ),
            (
                "relu6_feature_map",
                str(tuple(feature_map.shape)),
                lambda: custom_relu6(feature_map),
                lambda: torch.clamp(feature_map, min=0.0, max=6.0),
            ),
            (
                "linear_classifier",
                f"{tuple(flattened.shape)} -> ({batch_size}, 43)",
                lambda: custom_linear(flattened, classifier_weight, classifier_bias),
                lambda: F.linear(flattened, classifier_weight, classifier_bias),
            ),
            (
                "fused_linear_relu_benchmark",
                f"{tuple(flattened.shape)} -> ({batch_size}, 43)",
                lambda: custom_fused_linear_relu(flattened, classifier_weight, classifier_bias),
                lambda: torch.relu(F.linear(flattened, classifier_weight, classifier_bias)),
            ),
        )

        for name, shape, custom_fn, reference_fn in cases:
            results.append(
                _summarize_case(
                    name=name,
                    batch_size=batch_size,
                    shape=shape,
                    custom_fn=custom_fn,
                    reference_fn=reference_fn,
                    warmup=warmup,
                    repeat=repeat,
                )
            )

    return results


def write_csv(results: list[BenchmarkResult], output_csv: Path) -> None:
    """Write benchmark results to the M5/M7 table path."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(BenchmarkResult.__dataclass_fields__)
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations per case")
    parser.add_argument("--repeat", type=int, default=50, help="timed iterations per case")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    results = run_benchmarks(warmup=args.warmup, repeat=args.repeat)
    write_csv(results, args.output_csv)

    print(
        "kernel,batch_size,shape,reference,custom_mean_ms,custom_median_ms,"
        "custom_std_ms,reference_mean_ms,speedup_vs_reference,max_abs_error,mean_abs_error"
    )
    for result in results:
        print(
            f"{result.name},{result.batch_size},{result.shape},{result.reference},"
            f"{result.custom_mean_ms:.4f},{result.custom_median_ms:.4f},"
            f"{result.custom_std_ms:.4f},{result.reference_mean_ms:.4f},"
            f"{result.speedup_vs_reference:.4f},{result.max_abs_error:.6g},{result.mean_abs_error:.6g}"
        )
    print(f"Wrote benchmark CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
