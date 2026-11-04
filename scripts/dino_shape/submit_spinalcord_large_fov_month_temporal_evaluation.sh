#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
BANK_ROOT="${BANK_ROOT:-${STUDY_ROOT}/shared_probe_banks}"
export PROJECT_ROOT STUDY_ROOT BANK_ROOT
exec bash "${PROJECT_ROOT}/scripts/dino_shape/submit_spinalcord_large_fov_evaluation.sh"
