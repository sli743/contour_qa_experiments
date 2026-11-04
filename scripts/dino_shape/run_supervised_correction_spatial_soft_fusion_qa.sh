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
FUSION_MAX_EPOCHS="${FUSION_MAX_EPOCHS:-250}"
FUSION_PATIENCE="${FUSION_PATIENCE:-25}"
FUSION_MIN_DELTA="${FUSION_MIN_DELTA:-1e-4}"
FUSION_EARLYSTOP_FRACTION="${FUSION_EARLYSTOP_FRACTION:-0.20}"
FUSION_BATCH_SIZE="${FUSION_BATCH_SIZE:-8}"
FUSION_LR="${FUSION_LR:-1e-4}"
FUSION_WEIGHT_DECAY="${FUSION_WEIGHT_DECAY:-1e-3}"
FUSION_BASE_CHANNELS="${FUSION_BASE_CHANNELS:-8}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.25}"
FUSION_CV_FOLDS="${FUSION_CV_FOLDS:-5}"
QA_N_BOOT="${QA_N_BOOT:-2000}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CONFIG="${OUT}/config.yaml"
CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
TENSOR_ROOT="${OUT}/spatial_qa_tensor_cache"
SOFT_ROOT="${OUT}/soft_qa_features"
ANALYSIS_OUT="${OUT}/spatial_soft_fusion_qa_analysis"
HEAD_MODEL_DIR="${OUT}/spatial_soft_fusion_qa_models"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 3; }

for SPLIT in train validation test; do
  if [[ ! -f "${OUT}/evaluation/${SPLIT}_metrics.csv" ]]; then
    python -m evaluation.supervised_correction.eval_supervised_correction_threeclass \
      --config "${CONFIG}" --output-dir "${OUT}/evaluation"
    break
  fi
done

for SPLIT in train validation test; do
  if [[ ! -f "${TENSOR_ROOT}/${SPLIT}/index.csv" ]]; then
    python -m analysis.extract_supervised_correction_spatial_soft_qa_tensors \
      --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --split "${SPLIT}" \
      --local-radius "${LOCAL_RADIUS}" --boundary-radius "${BOUNDARY_RADIUS}" \
      --spatial-size "${SPATIAL_SIZE}" --output-dir "${TENSOR_ROOT}/${SPLIT}"
  else
    echo "Reusing cached spatial tensors for ${SPLIT}: ${TENSOR_ROOT}/${SPLIT}"
  fi
  if [[ ! -f "${SOFT_ROOT}/${SPLIT}/patient_action_score_summary.csv" ]]; then
    python -m analysis.extract_supervised_correction_soft_qa_features \
      --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --split "${SPLIT}" \
      --local-radius "${LOCAL_RADIUS}" --boundary-radius "${BOUNDARY_RADIUS}" \
      --output-csv "${SOFT_ROOT}/${SPLIT}/patient_action_score_summary.csv"
  else
    echo "Reusing compact soft QA features for ${SPLIT}: ${SOFT_ROOT}/${SPLIT}"
  fi
done

python -m analysis.train_supervised_correction_spatial_soft_fusion_qa \
  --tensor-root "${TENSOR_ROOT}" --soft-feature-root "${SOFT_ROOT}" \
  --evaluation-dir "${OUT}/evaluation" --output-dir "${ANALYSIS_OUT}" \
  --model-dir "${HEAD_MODEL_DIR}" --roi "${ROI}" --variant "${VARIANT}" \
  --max-epochs "${FUSION_MAX_EPOCHS}" --patience "${FUSION_PATIENCE}" \
  --min-delta "${FUSION_MIN_DELTA}" --earlystop-fraction "${FUSION_EARLYSTOP_FRACTION}" \
  --batch-size "${FUSION_BATCH_SIZE}" --lr "${FUSION_LR}" \
  --weight-decay "${FUSION_WEIGHT_DECAY}" --base-channels "${FUSION_BASE_CHANNELS}" \
  --dropout "${FUSION_DROPOUT}" --cv-folds "${FUSION_CV_FOLDS}" --n-boot "${QA_N_BOOT}"

python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_spatial_soft_fusion_qa_results_no_models.zip"

echo "Spatial + compact-soft fusion QA experiment complete: ${OUT}"
