#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
CONFIG="${CONFIG:-}"

if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: set CONFIG to the stochastic-region-dropout training YAML." >&2
  echo "Example:" >&2
  echo "  CONFIG=configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v3_symmetric_changed_loss.yaml \\" >&2
  echo "    bash scripts/dino_shape/run_one_patient_training_corruption_examples.sh" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

ARGS=(
  --config "${CONFIG}"
  --split "${SPLIT:-validation}"
)
if [[ -n "${PATIENT_ID:-}" ]]; then
  ARGS+=(--patient-id "${PATIENT_ID}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

python -m evaluation.visualization.export_one_patient_training_corruptions "${ARGS[@]}" "$@"
