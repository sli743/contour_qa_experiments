#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
export PROJECT_ROOT STUDY_ROOT REG_ROOT
exec bash "${PROJECT_ROOT}/scripts/dino_shape/submit_spinalcord_large_fov_ranking_classification.sh"
