#!/usr/bin/env bash
set -euo pipefail
ROI="${1:-Lung_L}"
VARIANT="${2:-large_decoder_ft_last2}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
FOLDS="${FOLDS:-0,1,2,3,4}"
LOCAL_RADIUS="${LOCAL_RADIUS:-6}"
FORCE_ACTION_DIAG="${FORCE_ACTION_DIAG:-0}"
QA_METHOD="${QA_METHOD:-joint_spatial_action_head}"
BAD_PERCENTILE="${BAD_PERCENTILE:-75}"
FOLDS_EXPORT="${FOLDS//,/:}"

cd "${PROJECT_ROOT}"
IFS=',' read -r -a FOLD_ARRAY <<< "${FOLDS}"
ACTION_JOB_IDS=()

for FOLD_RAW in "${FOLD_ARRAY[@]}"; do
  FOLD="$(echo "${FOLD_RAW}" | xargs)"
  [[ -n "${FOLD}" ]] || continue
  OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
  CONFIG="${OUT}/config.yaml"
  HISTORY="${OUT}/models/capacity_ft_ablation_${VARIANT}/logs/history.csv"
  CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
  ACTION_SENTINEL="${OUT}/action_score_diagnostic/action_score_discrimination.csv"

  [[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
  [[ -f "${HISTORY}" ]] || { echo "Missing training history: ${HISTORY}" >&2; exit 3; }
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 4; }

  if [[ "${FORCE_ACTION_DIAG}" == "1" || ! -f "${ACTION_SENTINEL}" ]]; then
    echo "Submitting validation action discrimination: fold=${FOLD}"
    JOB_ID=$(sbatch --parsable \
      --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${VARIANT},LOCAL_RADIUS=${LOCAL_RADIUS},FORCE_ACTION_DIAG=${FORCE_ACTION_DIAG},STUDY_ROOT=${STUDY_ROOT}" \
      -p hugo --account geoffrey_hugo \
      jobs/supervised_correction_fold_action_quality_diagnostic.job | cut -d';' -f1)
    ACTION_JOB_IDS+=("${JOB_ID}")
    echo "  action diagnostic job: ${JOB_ID}"
  else
    echo "Reusing fold ${FOLD} action discrimination: ${ACTION_SENTINEL}"
  fi
done

DEP_ARGS=()
if (( ${#ACTION_JOB_IDS[@]} > 0 )); then
  DEP_JOINED=$(IFS=:; echo "${ACTION_JOB_IDS[*]}")
  DEP_ARGS=(--dependency="afterok:${DEP_JOINED}")
fi

echo "Submitting fold training-quality aggregator"
AGG_JOB_ID=$(sbatch --parsable "${DEP_ARGS[@]}" \
  --export="ALL,ROI=${ROI},VARIANT=${VARIANT},FOLDS_EXPORT=${FOLDS_EXPORT},QA_METHOD=${QA_METHOD},BAD_PERCENTILE=${BAD_PERCENTILE},STUDY_ROOT=${STUDY_ROOT}" \
  -p hugo --account geoffrey_hugo \
  jobs/supervised_correction_fold_training_quality_diagnostic.job | cut -d';' -f1)

echo "Training-quality diagnostic job: ${AGG_JOB_ID}"
echo "Expected ZIP: ${STUDY_ROOT}/training_quality_diagnostics/${ROI}/${VARIANT}_training_quality_diagnostic.zip"
