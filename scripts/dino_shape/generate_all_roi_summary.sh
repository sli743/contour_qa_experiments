#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
python -m analysis.generate_all_roi_nsrqa_summary --study-root "${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
