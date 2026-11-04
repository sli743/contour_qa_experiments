#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-all_previous}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"
submit_one() {
  local variant="$1"
  sbatch --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${variant}" \
    -p hugo --account geoffrey_hugo \
    jobs/supervised_correction_capacity_finetune_qa_postprocess.job
}
if [[ "${VARIANT}" == "all_previous" ]]; then
  submit_one baseline
  submit_one large_decoder
  submit_one partial_ft_last1
else
  case "${VARIANT}" in
    baseline|large_decoder|partial_ft_last1|partial_ft_last2) submit_one "${VARIANT}" ;;
    *) echo "VARIANT must be all_previous, baseline, large_decoder, partial_ft_last1, or partial_ft_last2" >&2; exit 2 ;;
  esac
fi
