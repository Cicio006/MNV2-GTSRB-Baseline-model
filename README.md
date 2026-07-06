# MNV2-GTSRB-Baseline-model

MobileNetV2 baseline model for traffic sign classification on the GTSRB dataset.

## Overview

This repository contains a backend-independent MobileNetV2-style architecture adapted to 64x64 GTSRB images with 43 traffic sign classes. It includes CuPy backend layers, Triton kernel experiments, training utilities, and tests for the model and backend components.

## Repository Structure

- `src/models/` - MobileNetV2 model definition and sample inference helpers
- `src/backends/cupy/` - CuPy layer implementations
- `src/backends/triton/` - Triton kernels and wrappers
- `src/blocks/` - reusable MobileNet building blocks
- `src/layers/` - backend-independent layer interfaces
- `src/training/` - training-related utilities such as loss functions
- `tests/` - unit tests for layers, backends, kernels, and model contracts

## Local Data

The GTSRB dataset is intentionally not tracked in Git. Place local dataset files under:

```text
src/data/gtsrb/GTSRB/
```

Local dataset CSV files, checkpoints, caches, and generated artifacts are ignored by `.gitignore`.

## Running Tests

Install the required dependencies, then run:

```bash
pytest
```

Some backend tests require CUDA/CuPy or Triton support and may be skipped or fail on machines without a compatible GPU environment.
