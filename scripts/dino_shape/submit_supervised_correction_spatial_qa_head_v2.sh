#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
WHAT="${3:-large_decoder_ft_last2}"

case "${WHAT}" in
  large_decoder_ft_last2)
    VARIANTS=(large_decoder_ft_last2)
    ;;
  ft_pair)
    VARIANTS=(large_decoder_ft_last1 large_decoder_ft_last2)
    ;;
  core3)
    VARIANTS=(large_decoder large_decoder_ft_last1 large_decoder_ft_last2)
    ;;
  *)
    VARIANTS=("${WHAT}")
    ;;
esac

for VARIANT in "${VARIANTS[@]}"; do
  echo "Submitting spatial QA v2: ROI=${ROI} fold=${FOLD} variant=${VARIANT}"
  sbatch --export=ALL,ROI="${ROI}",FOLD="${FOLD}",VARIANT="${VARIANT}" \
    -p hugo --account=geoffrey_hugo \
    jobs/supervised_correction_spatial_qa_head_v2.job
done
