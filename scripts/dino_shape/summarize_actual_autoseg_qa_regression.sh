#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
QA_ROOT="${QA_ROOT:-${STUDY_ROOT}/qa_metric_regression}"
ROI="${1:?Usage: $0 ROI}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m analysis.summarize_actual_autoseg_qa_regression_folds --root "${QA_ROOT}/${ROI}"
