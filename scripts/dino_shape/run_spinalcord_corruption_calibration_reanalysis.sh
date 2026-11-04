#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${STUDY_ROOT}/spinalcord_corruption_calibration}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_corruption_calibration.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIBRATION_ROOT}/reanalysis_v2}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python analysis/reanalyze_spinalcord_corruption_calibration.py \
  --calibration-root "${CALIBRATION_ROOT}" \
  --analysis-config "${ANALYSIS_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo
echo "Reanalysis report: ${OUTPUT_DIR}/report.md"
