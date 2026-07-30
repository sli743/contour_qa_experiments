#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Bronchus_v1_eval.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLIT="${SPLIT:-test}"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -m evaluation.probes.eval_dino_shape_denoising_probe_sweep \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --out_subdir "probe_eval_sweep_${SPLIT}" \
  --inference_batch_size "${BATCH_SIZE}" \
  "$@"
