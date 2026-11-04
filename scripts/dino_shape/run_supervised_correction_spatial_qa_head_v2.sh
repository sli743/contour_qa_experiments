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
QA_V2_MAX_EPOCHS="${QA_V2_MAX_EPOCHS:-250}"
QA_V2_PATIENCE="${QA_V2_PATIENCE:-25}"
QA_V2_MIN_DELTA="${QA_V2_MIN_DELTA:-1e-4}"
QA_V2_EARLYSTOP_FRACTION="${QA_V2_EARLYSTOP_FRACTION:-0.20}"
QA_V2_BATCH_SIZE="${QA_V2_BATCH_SIZE:-8}"
QA_V2_LR="${QA_V2_LR:-1e-4}"
QA_V2_WEIGHT_DECAY="${QA_V2_WEIGHT_DECAY:-1e-3}"
QA_V2_BASE_CHANNELS="${QA_V2_BASE_CHANNELS:-8}"
QA_V2_DROPOUT="${QA_V2_DROPOUT:-0.30}"
QA_V2_CV_FOLDS="${QA_V2_CV_FOLDS:-5}"
QA_N_BOOT="${QA_N_BOOT:-2000}"

cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
CONFIG="${OUT}/config.yaml"
CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
TENSOR_ROOT="${OUT}/spatial_qa_tensor_cache"
ANALYSIS_OUT="${OUT}/spatial_qa_head_v2_analysis"
HEAD_MODEL_DIR="${OUT}/spatial_qa_head_v2_models"

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 3; }

# Reuse existing evaluation targets/tensor cache when present. Build only what is missing.
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
      --spatial-size "${SPATIAL_SIZE}" \
      --output-dir "${TENSOR_ROOT}/${SPLIT}"
  else
    echo "Reusing cached spatial tensors for ${SPLIT}: ${TENSOR_ROOT}/${SPLIT}"
  fi
done

python -m analysis.train_supervised_correction_spatial_qa_head_v2 \
  --tensor-root "${TENSOR_ROOT}" \
  --evaluation-dir "${OUT}/evaluation" \
  --output-dir "${ANALYSIS_OUT}" \
  --model-dir "${HEAD_MODEL_DIR}" \
  --roi "${ROI}" --variant "${VARIANT}" \
  --max-epochs "${QA_V2_MAX_EPOCHS}" --patience "${QA_V2_PATIENCE}" \
  --min-delta "${QA_V2_MIN_DELTA}" --earlystop-fraction "${QA_V2_EARLYSTOP_FRACTION}" \
  --batch-size "${QA_V2_BATCH_SIZE}" --lr "${QA_V2_LR}" \
  --weight-decay "${QA_V2_WEIGHT_DECAY}" --base-channels "${QA_V2_BASE_CHANNELS}" \
  --dropout "${QA_V2_DROPOUT}" --cv-folds "${QA_V2_CV_FOLDS}" --n-boot "${QA_N_BOOT}"

python scripts/dino_shape/package_supervised_correction_ablation_results.py \
  --input-dir "${OUT}" \
  --output-zip "${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}_spatial_qa_v2_results_no_models.zip"

echo "Single-target/direct-classification spatial QA experiment complete: ${OUT}"
