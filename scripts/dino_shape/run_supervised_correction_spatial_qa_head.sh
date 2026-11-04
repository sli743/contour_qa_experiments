#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Lung_L}"
FOLD="${2:-0}"
VARIANT="${3:-large_decoder_ft_last2}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-3}"
SPATIAL_SIZE="${SPATIAL_SIZE:-32,32,32}"
QA_HEAD_EPOCHS="${QA_HEAD_EPOCHS:-150}"
QA_HEAD_BATCH_SIZE="${QA_HEAD_BATCH_SIZE:-8}"
QA_HEAD_LR="${QA_HEAD_LR:-3e-4}"
QA_HEAD_WEIGHT_DECAY="${QA_HEAD_WEIGHT_DECAY:-1e-3}"
QA_HEAD_BASE_CHANNELS="${QA_HEAD_BASE_CHANNELS:-8}"
QA_HEAD_DROPOUT="${QA_HEAD_DROPOUT:-0.20}"
QA_HEAD_CV_FOLDS="${QA_HEAD_CV_FOLDS:-5}"
QA_N_BOOT="${QA_N_BOOT:-2000}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CONFIG="${OUT}/config.yaml"
CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
TENSOR_ROOT="${OUT}/spatial_qa_tensor_cache"
ANALYSIS_OUT="${OUT}/spatial_qa_head_analysis"
HEAD_MODEL_DIR="${OUT}/spatial_qa_head_models"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 3; }

# Ensure true input-vs-approved QA targets exist for train/validation/test.
python -m evaluation.supervised_correction.eval_supervised_correction_threeclass \
  --config "${CONFIG}" --output-dir "${OUT}/evaluation"

# The large correction model is run only once per split; downstream head training
# operates entirely on these compact fixed-size tensors.
for SPLIT in train validation test; do
  python -m analysis.extract_supervised_correction_spatial_soft_qa_tensors \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --split "${SPLIT}" \
    --local-radius "${LOCAL_RADIUS}" --boundary-radius "${BOUNDARY_RADIUS}" \
    --spatial-size "${SPATIAL_SIZE}" \
    --output-dir "${TENSOR_ROOT}/${SPLIT}"
done

python -m analysis.train_supervised_correction_spatial_qa_head \
  --tensor-root "${TENSOR_ROOT}" \
  --evaluation-dir "${OUT}/evaluation" \
  --output-dir "${ANALYSIS_OUT}" \
  --model-dir "${HEAD_MODEL_DIR}" \
  --roi "${ROI}" --variant "${VARIANT}" \
  --epochs "${QA_HEAD_EPOCHS}" --batch-size "${QA_HEAD_BATCH_SIZE}" \
  --lr "${QA_HEAD_LR}" --weight-decay "${QA_HEAD_WEIGHT_DECAY}" \
  --base-channels "${QA_HEAD_BASE_CHANNELS}" --dropout "${QA_HEAD_DROPOUT}" \
  --cv-folds "${QA_HEAD_CV_FOLDS}" --n-boot "${QA_N_BOOT}"

python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_spatial_qa_results_no_models.zip"

echo "Learned spatial QA-head experiment complete: ${OUT}"
