#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-all}"
ACTION_WEIGHT="${4:-4.0}"
ENCODER_LR="${5:-1e-6}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
cd "${PROJECT_ROOT}"

submit_one() {
  local variant="$1"
  sbatch --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${variant},ACTION_WEIGHT=${ACTION_WEIGHT},ENCODER_LR=${ENCODER_LR}" \
    -p hugo --account geoffrey_hugo \
    jobs/supervised_correction_capacity_finetune_ablation.job
}

if [[ "${VARIANT}" == "all" ]]; then
  submit_one baseline
  submit_one large_decoder
  submit_one partial_ft_last1
  submit_one partial_ft_last2
  submit_one large_decoder_ft_last1
  submit_one large_decoder_ft_last2
elif [[ "${VARIANT}" == "large_decoder_ft" ]]; then
  # The frozen large_decoder result is the reference from the previous ablation.
  # Submit only the two new encoder-adaptation runs by default.
  submit_one large_decoder_ft_last1
  submit_one large_decoder_ft_last2
else
  case "${VARIANT}" in
    baseline|large_decoder|partial_ft_last1|partial_ft_last2|large_decoder_ft_last1|large_decoder_ft_last2)
      submit_one "${VARIANT}"
      ;;
    *)
      echo "VARIANT must be all, large_decoder_ft, baseline, large_decoder, partial_ft_last1, partial_ft_last2, large_decoder_ft_last1, or large_decoder_ft_last2" >&2
      exit 2
      ;;
  esac
fi
