#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v2_global_boundary.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# PyTorch 2.0 in the 3DINO environment does not understand expandable_segments.
unset PYTORCH_CUDA_ALLOC_CONF || true
python -m analysis.diagnose_v2_directional_asymmetry \
  --config "${CONFIG}" \
  --split test \
  --offset-mm 1.5 2 3 4 5 6 \
  "$@"
