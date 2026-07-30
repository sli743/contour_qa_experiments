#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Bronchus_v1_eval.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLIT="${SPLIT:-test}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-probe_visualization_${SPLIT}}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m evaluation.probes.export_dino_shape_denoising_probe_contours \
  --config "${CONFIG}" --split "${SPLIT}" \
  --output_subdir "${OUTPUT_SUBDIR}" \
  --inference_batch_size "${BATCH_SIZE}" \
  --layout flat --save_metrics "$@"
