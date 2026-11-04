#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${1:?Usage: $0 ROI [FOLD|all] [CORRECTION_MODE]}"
FOLD="${2:-all}"
CORRECTION_MODE="${3:-auto}"
cd "${PROJECT_ROOT}"

submit_one() {
  local fold="$1"
  sbatch --export=ALL,ROI="${ROI}",FOLD="${fold}",CORRECTION_MODE="${CORRECTION_MODE}" -p hugo --account geoffrey_hugo \
    jobs/actual_autoseg_qa_regression.job
}

if [[ "${FOLD}" == "all" ]]; then
  for fold in 0 1 2 3 4; do submit_one "${fold}"; done
else
  submit_one "${FOLD}"
fi
