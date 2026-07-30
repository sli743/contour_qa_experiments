#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Bronchus_probe_eval.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLIT="${SPLIT:-test}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-probe_visualization_${SPLIT}_full_benchmark}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Same full benchmark corruption generator and severity grids used by the
# denoising and random-morphology visualization exporters.
# The exporter writes the flat ProbeContourComparer-compatible structure.
python -m evaluation.probes.export_dino_shape_denoising_probe_contours \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --sample_ratio "${SAMPLE_RATIO}" \
  --inference_batch_size "${BATCH_SIZE}" \
  --output_subdir "${OUTPUT_SUBDIR}" \
  "$@"
