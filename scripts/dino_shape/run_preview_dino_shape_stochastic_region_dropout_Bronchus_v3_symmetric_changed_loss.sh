#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Bronchus_v3_symmetric_changed_loss.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m evaluation.visualization.export_fixed_global_boundary_offset_examples \
  --config "${CONFIG}" \
  --split validation \
  --max-cases "${MAX_CASES:-8}" \
  --offset-mm 1.5 \
  --offset-mm 2.0 \
  --offset-mm 2.5 \
  --offset-mm 3.0 \
  "$@"
