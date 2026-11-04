#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"
JOB_ID=$(sbatch \
  -p hugo \
  --account geoffrey_hugo \
  --export="ALL,ROI=${ROI},FOLD=${FOLD}" \
  --parsable \
  jobs/supervised_correction_residual_v4.job)
echo "Submitted directional residual v4: ROI=${ROI} fold=${FOLD} job=${JOB_ID}"
