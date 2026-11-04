#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-baseline}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
ENCODER_LR="${ENCODER_LR:-1e-6}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"
QA_N_BOOT="${QA_N_BOOT:-2000}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG=$(python scripts/dino_shape/prepare_supervised_correction_capacity_finetune_ablation.py \
  --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}" \
  --variant "${VARIANT}" --action-weight "${ACTION_WEIGHT}" --encoder-lr "${ENCODER_LR}" | tail -n 1)
OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"

echo "Config: ${CONFIG}"
echo "ROI=${ROI} fold=${FOLD} variant=${VARIANT} action_weight=${ACTION_WEIGHT} encoder_lr=${ENCODER_LR}"
python -m training.train_supervised_autoseg_correction_threeclass --config "${CONFIG}" --fresh
python -m evaluation.supervised_correction.eval_supervised_correction_threeclass \
  --config "${CONFIG}" --output-dir "${OUT}/evaluation"
python -m analysis.diagnose_supervised_correction_action_scores \
  --config "${CONFIG}" --split validation --local-radius "${LOCAL_RADIUS}" \
  --output-dir "${OUT}/action_score_diagnostic"
python -m analysis.analyze_supervised_correction_qa_signal \
  --evaluation-dir "${OUT}/evaluation" --output-dir "${OUT}/qa_signal_analysis" \
  --roi "${ROI}" --variant "${VARIANT}" --n-boot "${QA_N_BOOT}"
python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_results_no_models.zip"
echo "Done. Results: ${OUT}"
