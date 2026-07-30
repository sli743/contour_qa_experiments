#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Bronchus_v1.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Benchmark-leakage upper-bound experiment:
# train on the same corruption families used in benchmark evaluation.
python -m training.train_dino_shape_denoising_decoder   --config "${CONFIG}"   "$@"
