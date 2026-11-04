#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-baseline}"
QA_N_BOOT="${QA_N_BOOT:-2000}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CONFIG="${OUT}/config.yaml"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth" ]]; then
  echo "Missing best checkpoint for ${VARIANT}: ${OUT}" >&2
  exit 3
fi
python -m evaluation.supervised_correction.eval_supervised_correction_threeclass \
  --config "${CONFIG}" --output-dir "${OUT}/evaluation"
python -m analysis.analyze_supervised_correction_qa_signal \
  --evaluation-dir "${OUT}/evaluation" --output-dir "${OUT}/qa_signal_analysis" \
  --roi "${ROI}" --variant "${VARIANT}" --n-boot "${QA_N_BOOT}"
python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_results_no_models.zip"
echo "QA postprocessing complete: ${OUT}"
