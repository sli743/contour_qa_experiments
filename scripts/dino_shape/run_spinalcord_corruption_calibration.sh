#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_corruption_calibration.yaml}"
NSRQA_CONFIG="${NSRQA_CONFIG:-${STUDY_ROOT}/configs/SpinalCord/nsrqa.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${STUDY_ROOT}/spinalcord_corruption_calibration}"
TRAINING_BANK_SAMPLES_PER_CASE="${TRAINING_BANK_SAMPLES_PER_CASE:-20}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${SPLIT_CONFIG}" ]]; then
  echo "ERROR: temporal split config not found: ${SPLIT_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${NSRQA_CONFIG}" ]]; then
  echo "ERROR: prepared NSR-QA config not found: ${NSRQA_CONFIG}" >&2
  echo "Run prepare_spinalcord_large_fov_month_temporal_study.sh first." >&2
  exit 1
fi

python analysis/analyze_spinalcord_corruption_calibration.py \
  --config "${ANALYSIS_CONFIG}" \
  --raw-data-root "${RAW_DATA_ROOT}" \
  --split-config "${SPLIT_CONFIG}" \
  --study-root "${STUDY_ROOT}" \
  --nsrqa-config "${NSRQA_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --training-bank-samples-per-case "${TRAINING_BANK_SAMPLES_PER_CASE}" \
  "$@"

echo
echo "Corruption calibration report: ${OUTPUT_DIR}/report.md"
