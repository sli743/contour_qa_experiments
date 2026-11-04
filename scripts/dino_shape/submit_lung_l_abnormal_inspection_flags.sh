#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/lung_l_abnormal_inspection_flags.yaml}"
JOB="${PROJECT_ROOT}/jobs/lung_l_abnormal_inspection_flags.job"

cd "${PROJECT_ROOT}"

sbatch \
  -p hugo \
  --account=geoffrey_hugo \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG=${CONFIG}" \
  "${JOB}"
