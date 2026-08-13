#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
TASKS="${STUDY_ROOT}/manifests/training_tasks.tsv"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
[[ -s "${TASKS}" ]] || { echo "Missing ${TASKS}; run prepare/examples first." >&2; exit 2; }
n=$(( $(wc -l < "${TASKS}") - 1 ))
(( n > 0 )) || { echo "No training tasks." >&2; exit 2; }
last=$((n-1))
cd "${PROJECT_ROOT}"
echo "Submitting ${n} four-model training tasks, max concurrent=${MAX_CONCURRENT}"
# Partition/account are deliberately explicit here AND in the job file.
sbatch -p hugo --account geoffrey_hugo --array="0-${last}%${MAX_CONCURRENT}" jobs/all_roi_four_model_train_array.job
