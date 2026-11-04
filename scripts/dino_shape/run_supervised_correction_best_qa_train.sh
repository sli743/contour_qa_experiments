#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:?ROI required}"
FOLD="${2:?fold required}"
VARIANT="${3:-large_decoder_ft_last2}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
ENCODER_LR="${ENCODER_LR:-1e-6}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG=$(python scripts/dino_shape/prepare_supervised_correction_capacity_finetune_ablation.py \
  --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}" \
  --variant "${VARIANT}" --action-weight "${ACTION_WEIGHT}" --encoder-lr "${ENCODER_LR}" | tail -n 1)

echo "Training fixed all-ROI QA representation"
echo "ROI=${ROI} fold=${FOLD} variant=${VARIANT}"
echo "Config=${CONFIG}"
python -m training.train_supervised_autoseg_correction_threeclass --config "${CONFIG}" --fresh
