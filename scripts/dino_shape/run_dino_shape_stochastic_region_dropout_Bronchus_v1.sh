#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Bronchus_v1.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
python -m training.train_dino_shape_stochastic_region_dropout_decoder --config "${CONFIG}" "$@"
