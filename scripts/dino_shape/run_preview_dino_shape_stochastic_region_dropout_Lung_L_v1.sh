#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v1.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m evaluation.visualization.export_stochastic_region_dropout_examples --config "${CONFIG}" "$@"
