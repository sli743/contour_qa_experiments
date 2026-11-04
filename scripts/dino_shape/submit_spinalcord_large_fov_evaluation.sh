#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study}"
BANK_ROOT="${BANK_ROOT:-${STUDY_ROOT}/shared_probe_banks}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
# Larger Z doubles the token count relative to the original SpinalCord study.
# Use a conservative probe-inference batch size by default.
BATCH_SIZE="${BATCH_SIZE:-4}"
export PROJECT_ROOT STUDY_ROOT BANK_ROOT MAX_CONCURRENT BATCH_SIZE
cd "${PROJECT_ROOT}"
bash scripts/dino_shape/submit_all_roi_evaluation.sh
