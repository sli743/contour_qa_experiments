#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-large_decoder_ft_last2}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-3}"
QA_N_BOOT="${QA_N_BOOT:-2000}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CONFIG="${OUT}/config.yaml"
CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
SOFT_ROOT="${OUT}/soft_qa_features"
ANALYSIS_OUT="${OUT}/soft_qa_readout_analysis"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 3; }

# Ensure hard-output QA metrics exist for exactly the same checkpoint/splits.
python -m evaluation.supervised_correction.eval_supervised_correction_threeclass \
  --config "${CONFIG}" --output-dir "${OUT}/evaluation"

for SPLIT in train validation test; do
  python -m analysis.extract_supervised_correction_soft_qa_features \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --split "${SPLIT}" \
    --local-radius "${LOCAL_RADIUS}" --boundary-radius "${BOUNDARY_RADIUS}" \
    --output-csv "${SOFT_ROOT}/${SPLIT}/patient_action_score_summary.csv"
done

python -m analysis.analyze_supervised_correction_soft_qa_readout \
  --evaluation-dir "${OUT}/evaluation" \
  --soft-feature-root "${SOFT_ROOT}" \
  --output-dir "${ANALYSIS_OUT}" \
  --roi "${ROI}" --variant "${VARIANT}" --n-boot "${QA_N_BOOT}"

python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_soft_qa_results_no_models.zip"

echo "Soft QA readout complete: ${OUT}"
