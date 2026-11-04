#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study}"
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
export PROJECT_ROOT STUDY_ROOT MAX_CONCURRENT
cd "${PROJECT_ROOT}"
# The generic launcher submits the four SpinalCord tasks from this study's manifest.
bash scripts/dino_shape/submit_all_roi_training.sh
