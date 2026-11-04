#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${STUDY_ROOT}/spinalcord_corruption_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIBRATION_ROOT}/proposed_distribution_validation}"
NSRQA_CONFIG="${NSRQA_CONFIG:-${STUDY_ROOT}/configs/SpinalCord/nsrqa.yaml}"
PROPOSAL_CONFIG="${PROPOSAL_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_proposed_corruption_distribution.yaml}"
WORKERS="${WORKERS:-8}"
SAMPLES_PER_CASE="${SAMPLES_PER_CASE:-0}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

python analysis/analyze_spinalcord_proposed_corruption_distribution.py \
  --calibration-root "${CALIBRATION_ROOT}" \
  --study-root "${STUDY_ROOT}" \
  --nsrqa-config "${NSRQA_CONFIG}" \
  --proposal-config "${PROPOSAL_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --samples-per-case "${SAMPLES_PER_CASE}" \
  "$@"
