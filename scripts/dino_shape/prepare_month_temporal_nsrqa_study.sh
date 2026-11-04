#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python scripts/dino_shape/prepare_month_temporal_nsrqa_study.py \
  --split-config "${SPLIT_CONFIG}" \
  --study-root "${STUDY_ROOT}" \
  "$@"

echo
echo "Generating the same standardized NSR-QA corruption examples used by the August-13 workflow."
STUDY_ROOT="${STUDY_ROOT}" bash scripts/dino_shape/generate_all_roi_training_examples.sh

echo
echo "Inspect before training: ${STUDY_ROOT}/training_corruption_examples"
