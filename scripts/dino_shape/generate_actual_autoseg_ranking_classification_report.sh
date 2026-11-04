#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
ACTUAL_ROOT="${ACTUAL_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
OUTPUT_DIR="${OUTPUT_DIR:-${ACTUAL_ROOT}/ranking_classification_report}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"

python analysis/generate_actual_autoseg_ranking_classification_report.py \
  --root "${ACTUAL_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo "Report: ${OUTPUT_DIR}/report.md"
