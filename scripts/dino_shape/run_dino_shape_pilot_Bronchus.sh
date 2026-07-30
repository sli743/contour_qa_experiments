#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_pilot_Bronchus.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
MAX_CASES="${MAX_CASES:-6}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# 1. Train a decoder on clean approved contours.
#    The pretrained 3DINO encoder remains frozen.
python -m training.train_dino_shape_decoder   --config "${CONFIG}"

# 2. Export clean test reconstructions.
python -m evaluation.eval_dino_shape_clean   --config "${CONFIG}"   --split test

# 3. Export a small visual probe set.
python -m evaluation.probes.eval_dino_shape_probes_visual   --config "${CONFIG}"   --split test   --max_cases "${MAX_CASES}"   "$@"
