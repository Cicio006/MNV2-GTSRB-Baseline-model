#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Neither '$PYTHON_BIN' nor python3 was found." >&2
        exit 127
    fi
fi

WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-50}"
OUTPUT_CSV="${OUTPUT_CSV:-results/tables/kernel_benchmarks.csv}"

PYTHONPATH=. "$PYTHON_BIN" -m src.benchmarks.benchmark_kernels \
    --warmup "$WARMUP" \
    --repeat "$REPEAT" \
    --output-csv "$OUTPUT_CSV"
