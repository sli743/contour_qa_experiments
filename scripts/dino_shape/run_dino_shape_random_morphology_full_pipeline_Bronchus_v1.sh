#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
export PROJECT_ROOT
TRAIN_CONFIG="${TRAIN_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Bronchus_v1.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Bronchus_v1_eval.yaml}"
for required in "$TRAIN_CONFIG" "$EVAL_CONFIG"; do
  [[ -f "$required" ]] || { echo "ERROR: missing config: $required" >&2; exit 2; }
done
echo "[1/4] Training Bronchus random-morphology DAE"
CONFIG="$TRAIN_CONFIG" bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_Bronchus_v1.sh"
echo "[2/4] Evaluating fixed probe benchmark"
CONFIG="$EVAL_CONFIG" bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_probe_sweep_Bronchus_v1.sh"
echo "[3/4] Exporting testing-patient case visualizations"
CONFIG="$EVAL_CONFIG" bash "${PROJECT_ROOT}/scripts/dino_shape/run_export_dino_shape_random_morphology_probe_contours_Bronchus_v1.sh"
echo "[4/4] Three-model ROC comparison"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_compare_dino_shape_three_models_roc_Bronchus.sh"
