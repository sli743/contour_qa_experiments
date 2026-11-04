#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-large_decoder_ft_last2}"
FORCE_READOUTS="${FORCE_READOUTS:-0}"

cd "${PROJECT_ROOT}"
OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
[[ -f "${CHECKPOINT}" ]] || { echo "Missing trained correction checkpoint: ${CHECKPOINT}" >&2; exit 3; }

run_if_missing() {
  local sentinel="$1"
  shift
  if [[ "${FORCE_READOUTS}" == "1" || ! -f "${sentinel}" ]]; then
    echo "Running: $*"
    "$@"
  else
    echo "Reusing existing result: ${sentinel}"
  fi
}

# 1) Handcrafted soft local / local+boundary readouts.
run_if_missing \
  "${OUT}/soft_qa_readout_analysis/soft_qa_regression_summary.csv" \
  bash scripts/dino_shape/run_supervised_correction_soft_qa_readout.sh \
  "${ROI}" "${FOLD}" "${VARIANT}"

# 2) Joint spatial 3-target head. This is the current strongest binary readout.
run_if_missing \
  "${OUT}/spatial_qa_head_analysis/spatial_qa_regression_summary.csv" \
  bash scripts/dino_shape/run_supervised_correction_spatial_qa_head.sh \
  "${ROI}" "${FOLD}" "${VARIANT}"

# 3) Spatial + compact-soft fusion. Kept as a candidate because fold 0 showed
# competitive ASSD severity ranking, even though it was not the best classifier.
run_if_missing \
  "${OUT}/spatial_soft_fusion_qa_analysis/fusion_qa_regression_summary.csv" \
  bash scripts/dino_shape/run_supervised_correction_spatial_soft_fusion_qa.sh \
  "${ROI}" "${FOLD}" "${VARIANT}"

python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_best_qa_results_no_models.zip"

echo "Best-QA readouts complete: ROI=${ROI} fold=${FOLD} variant=${VARIANT}"
