#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/lung_l_corruption_severity_calibration.yaml}"
cd "${PROJECT_ROOT}"

sbatch \
  -p hugo \
  --account=geoffrey_hugo \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG=${CONFIG}" \
  jobs/lung_l_corruption_severity_calibration.job
