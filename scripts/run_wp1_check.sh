#!/bin/bash

#SBATCH --job-name=wp1_triton_check
#SBATCH --output=results/wp1/wp1_triton_check_%j.out
#SBATCH --error=results/wp1/wp1_triton_check_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH -p exercise-gpu

set -euo pipefail

echo "=== Node ==="
hostname

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
fi

echo "=== Working directory ==="
pwd

echo "=== GPU ==="
nvidia-smi

echo "=== Activate conda environment if available ==="
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "No conda command or standard conda.sh found; continuing with system Python."
fi

if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV_NAME:-eml}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Neither '$PYTHON_BIN' nor python3 was found after environment setup." >&2
        echo "Set PYTHON_BIN=/path/to/python or fix conda activation before submitting." >&2
        exit 127
    fi
fi

echo "=== Python ==="
which "$PYTHON_BIN"
"$PYTHON_BIN" --version

echo "=== Run environment check ==="
python src/backends/triton/utils/check_env.py
