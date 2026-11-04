#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-bbox_midpoint}"
export PROJECT_ROOT STUDY_ROOT REG_ROOT MAX_CONCURRENT AUTOSEG_CENTER_METHOD
cd "${PROJECT_ROOT}"
# The revised study uses bbox-midpoint localization for autoseg_centered mode.
# This preserves deployment-safe candidate-only localization while avoiding the
# foreground-centroid shift that can hide S/I extent errors in long structures.
bash scripts/dino_shape/submit_actual_autoseg_regression.sh
