#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m analysis.summarize_actual_autoseg_ranking_classification --root "${REG_ROOT}"
