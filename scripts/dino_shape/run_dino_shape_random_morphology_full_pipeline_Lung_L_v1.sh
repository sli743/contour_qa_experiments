#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
export PROJECT_ROOT

TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Lung_L_v1.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Lung_L_v1_eval.yaml}"

for required in "${TRAIN_CONFIG}" "${EVAL_CONFIG}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required config does not exist: ${required}" >&2
    exit 2
  fi
done

echo "[1/3] Training random-morphology DAE"
CONFIG="${TRAIN_CONFIG}" \
  bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_Lung_L_v1.sh"

echo "[2/3] Evaluating on the fixed probe benchmark"
CONFIG="${EVAL_CONFIG}" \
  bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_probe_sweep_Lung_L_v1.sh"

echo "[3/3] Building three-model ROC reports"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_compare_dino_shape_three_models_roc_Lung_L.sh"
