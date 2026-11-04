#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Lung_L}"
VARIANT="${2:-large_decoder_ft_last2}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
FOLDS="${FOLDS:-0,1,2,3,4}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
ENCODER_LR="${ENCODER_LR:-1e-6}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_READOUTS="${FORCE_READOUTS:-0}"
QA_N_BOOT="${QA_N_BOOT:-2000}"
FOLDS_EXPORT="${FOLDS//,/:}"

cd "${PROJECT_ROOT}"
IFS=',' read -r -a FOLD_ARRAY <<< "${FOLDS}"
READOUT_JOB_IDS=()

for FOLD_RAW in "${FOLD_ARRAY[@]}"; do
  FOLD="$(echo "${FOLD_RAW}" | xargs)"
  [[ -n "${FOLD}" ]] || continue
  BASE_CONFIG="${STUDY_ROOT}/configs/${ROI}/fold_${FOLD}.yaml"
  OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
  CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
  [[ -f "${BASE_CONFIG}" ]] || { echo "Missing base fold config: ${BASE_CONFIG}" >&2; exit 2; }

  TRAIN_JOB_ID=""
  if [[ "${FORCE_TRAIN}" == "1" || ! -f "${CHECKPOINT}" ]]; then
    echo "Submitting correction training: ROI=${ROI} fold=${FOLD} variant=${VARIANT}"
    TRAIN_JOB_ID=$(sbatch --parsable \
      --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${VARIANT},ACTION_WEIGHT=${ACTION_WEIGHT},ENCODER_LR=${ENCODER_LR}" \
      -p hugo --account geoffrey_hugo \
      jobs/supervised_correction_capacity_finetune_ablation.job | cut -d';' -f1)
    echo "  training job: ${TRAIN_JOB_ID}"
  else
    echo "Reusing trained correction model: ${CHECKPOINT}"
  fi

  SOFT_SENTINEL="${OUT}/soft_qa_readout_analysis/soft_qa_regression_summary.csv"
  SPATIAL_SENTINEL="${OUT}/spatial_qa_head_analysis/spatial_qa_regression_summary.csv"
  FUSION_SENTINEL="${OUT}/spatial_soft_fusion_qa_analysis/fusion_qa_regression_summary.csv"
  if [[ -z "${TRAIN_JOB_ID}" && "${FORCE_READOUTS}" != "1" && -f "${SOFT_SENTINEL}" && -f "${SPATIAL_SENTINEL}" && -f "${FUSION_SENTINEL}" ]]; then
    echo "Reusing completed QA readouts: ROI=${ROI} fold=${FOLD}"
    continue
  fi

  READOUT_FORCE="${FORCE_READOUTS}"
  if [[ -n "${TRAIN_JOB_ID}" ]]; then
    # A newly trained/retrained representation must be followed by fresh readouts;
    # never silently aggregate stale QA outputs from an older checkpoint.
    READOUT_FORCE=1
  fi

  DEP_ARGS=()
  if [[ -n "${TRAIN_JOB_ID}" ]]; then
    DEP_ARGS=(--dependency="afterok:${TRAIN_JOB_ID}")
  fi
  echo "Submitting best QA readouts: ROI=${ROI} fold=${FOLD} variant=${VARIANT}"
  READOUT_JOB_ID=$(sbatch --parsable "${DEP_ARGS[@]}" \
    --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${VARIANT},FORCE_READOUTS=${READOUT_FORCE}" \
    -p hugo --account geoffrey_hugo \
    jobs/supervised_correction_best_qa_readouts.job | cut -d';' -f1)
  READOUT_JOB_IDS+=("${READOUT_JOB_ID}")
  echo "  readout job: ${READOUT_JOB_ID}"
done

AGG_DEP_ARGS=()
if (( ${#READOUT_JOB_IDS[@]} > 0 )); then
  DEP_JOINED=$(IFS=:; echo "${READOUT_JOB_IDS[*]}")
  AGG_DEP_ARGS=(--dependency="afterok:${DEP_JOINED}")
fi

echo "Submitting cross-fold aggregator: ROI=${ROI} variant=${VARIANT} folds=${FOLDS}"
AGG_JOB_ID=$(sbatch --parsable "${AGG_DEP_ARGS[@]}" \
  --export="ALL,ROI=${ROI},VARIANT=${VARIANT},FOLDS_EXPORT=${FOLDS_EXPORT},QA_N_BOOT=${QA_N_BOOT},STUDY_ROOT=${STUDY_ROOT}" \
  -p hugo --account geoffrey_hugo \
  jobs/supervised_correction_best_qa_aggregate.job | cut -d';' -f1)

echo "Cross-fold aggregation job: ${AGG_JOB_ID}"
echo "Expected summary ZIP: ${STUDY_ROOT}/best_qa_5fold/${ROI}/${VARIANT}_best_qa_5fold_results_no_models.zip"
