#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
VARIANT="${VARIANT:-large_decoder_ft_last2}"
FOLDS="${FOLDS:-0,1,2,3,4}"
ROIS="${ROIS:-}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
ENCODER_LR="${ENCODER_LR:-1e-6}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_READOUTS="${FORCE_READOUTS:-0}"
QA_N_BOOT="${QA_N_BOOT:-2000}"
DRY_RUN="${DRY_RUN:-0}"
ACTIVE_MANIFEST="${ALLROI_ROOT}/manifests/active_rois.tsv"

cd "${PROJECT_ROOT}"
[[ -f "${ACTIVE_MANIFEST}" ]] || { echo "Missing active ROI manifest: ${ACTIVE_MANIFEST}" >&2; exit 2; }

if [[ -n "${ROIS}" ]]; then
  IFS=',' read -r -a ROI_ARRAY <<< "${ROIS}"
else
  mapfile -t ROI_ARRAY < <(awk -F '\t' 'NR>1 {print $3}' "${ACTIVE_MANIFEST}")
fi
IFS=',' read -r -a FOLD_ARRAY <<< "${FOLDS}"
(( ${#ROI_ARRAY[@]} > 0 )) || { echo "No ROIs selected" >&2; exit 2; }

# Strict preflight: this sweep is intended to cover every selected ROI with all five
# outer folds. Do not silently drop an ROI because preprocessing/config generation is incomplete.
for ROI_RAW in "${ROI_ARRAY[@]}"; do
  ROI="$(echo "${ROI_RAW}" | xargs)"
  [[ -n "${ROI}" ]] || continue
  awk -F '\t' -v r="${ROI}" 'NR>1 && $3==r {found=1} END{exit !found}' "${ACTIVE_MANIFEST}" || {
    echo "ROI is not active in ${ACTIVE_MANIFEST}: ${ROI}" >&2; exit 2;
  }
  for FOLD_RAW in "${FOLD_ARRAY[@]}"; do
    FOLD="$(echo "${FOLD_RAW}" | xargs)"
    CFG="${STUDY_ROOT}/configs/${ROI}/fold_${FOLD}.yaml"
    [[ -f "${CFG}" ]] || {
      echo "Missing supervised-correction fold config: ${CFG}" >&2
      echo "Prepare actual-autoseg data/configs first, e.g.:" >&2
      echo "  bash scripts/dino_shape/submit_supervised_autoseg_correction.sh preprocess" >&2
      echo "  bash scripts/dino_shape/submit_supervised_autoseg_correction.sh prepare" >&2
      exit 2
    }
  done
done

submit_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'; printf ' %q' "$@"; printf '\n'
    echo "DRYRUN"
  else
    "$@"
  fi
}

ROI_AGG_JOB_IDS=()
SELECTED_ROIS=()
for ROI_RAW in "${ROI_ARRAY[@]}"; do
  ROI="$(echo "${ROI_RAW}" | xargs)"
  [[ -n "${ROI}" ]] || continue
  SELECTED_ROIS+=("${ROI}")
  echo "=== ROI ${ROI} ==="
  READOUT_JOB_IDS=()

  for FOLD_RAW in "${FOLD_ARRAY[@]}"; do
    FOLD="$(echo "${FOLD_RAW}" | xargs)"
    [[ -n "${FOLD}" ]] || continue
    OUT="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}"
    CHECKPOINT="${OUT}/models/capacity_ft_ablation_${VARIANT}/checkpoints/best_decoder.pth"
    SPATIAL_SENTINEL="${OUT}/spatial_qa_head_analysis/spatial_qa_regression_summary.csv"

    TRAIN_JOB_ID=""
    if [[ "${FORCE_TRAIN}" == "1" || ! -f "${CHECKPOINT}" ]]; then
      echo "Submitting training: ROI=${ROI} fold=${FOLD}"
      RAW=$(submit_cmd sbatch --parsable \
        --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${VARIANT},ACTION_WEIGHT=${ACTION_WEIGHT},ENCODER_LR=${ENCODER_LR},STUDY_ROOT=${STUDY_ROOT}" \
        -p hugo --account geoffrey_hugo \
        jobs/supervised_correction_best_qa_train.job)
      TRAIN_JOB_ID="$(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
      echo "  training job: ${TRAIN_JOB_ID}"
    else
      echo "Reusing checkpoint: ${CHECKPOINT}"
    fi

    if [[ -z "${TRAIN_JOB_ID}" && "${FORCE_READOUTS}" != "1" && -f "${SPATIAL_SENTINEL}" ]]; then
      echo "Reusing joint spatial QA: ROI=${ROI} fold=${FOLD}"
      continue
    fi

    DEP_ARGS=()
    if [[ -n "${TRAIN_JOB_ID}" && "${TRAIN_JOB_ID}" != "DRYRUN" ]]; then
      DEP_ARGS=(--dependency="afterok:${TRAIN_JOB_ID}")
    fi
    echo "Submitting joint spatial QA: ROI=${ROI} fold=${FOLD}"
    RAW=$(submit_cmd sbatch --parsable "${DEP_ARGS[@]}" \
      --export="ALL,ROI=${ROI},FOLD=${FOLD},VARIANT=${VARIANT},STUDY_ROOT=${STUDY_ROOT},QA_N_BOOT=${QA_N_BOOT}" \
      -p hugo --account geoffrey_hugo \
      jobs/supervised_correction_spatial_qa_head.job)
    READOUT_JOB_ID="$(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
    if [[ "${READOUT_JOB_ID}" != "DRYRUN" ]]; then READOUT_JOB_IDS+=("${READOUT_JOB_ID}"); fi
    echo "  readout job: ${READOUT_JOB_ID}"
  done

  ROI_DEP_ARGS=()
  if (( ${#READOUT_JOB_IDS[@]} > 0 )); then
    DEP_JOINED=$(IFS=:; echo "${READOUT_JOB_IDS[*]}")
    ROI_DEP_ARGS=(--dependency="afterok:${DEP_JOINED}")
  fi
  FOLDS_EXPORT="${FOLDS//,/:}"
  ROI_AGG_OUT="${STUDY_ROOT}/joint_spatial_qa_5fold/${ROI}/${VARIANT}"
  ROI_AGG_ZIP="${STUDY_ROOT}/joint_spatial_qa_5fold/${ROI}/${VARIANT}_joint_spatial_5fold_results_no_models.zip"
  echo "Submitting five-fold joint-spatial aggregate: ROI=${ROI}"
  RAW=$(submit_cmd sbatch --parsable "${ROI_DEP_ARGS[@]}" \
    --export="ALL,ROI=${ROI},VARIANT=${VARIANT},FOLDS_EXPORT=${FOLDS_EXPORT},QA_N_BOOT=${QA_N_BOOT},STUDY_ROOT=${STUDY_ROOT},METHODS_EXPORT=joint_spatial_action_head,OUTPUT_DIR=${ROI_AGG_OUT},OUTPUT_ZIP=${ROI_AGG_ZIP}" \
    -p hugo --account geoffrey_hugo \
    jobs/supervised_correction_best_qa_aggregate.job)
  ROI_AGG_JOB_ID="$(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
  if [[ "${ROI_AGG_JOB_ID}" != "DRYRUN" ]]; then ROI_AGG_JOB_IDS+=("${ROI_AGG_JOB_ID}"); fi
  echo "  ROI aggregate job: ${ROI_AGG_JOB_ID}"
done

ALLROI_DEP_ARGS=()
if (( ${#ROI_AGG_JOB_IDS[@]} > 0 )); then
  DEP_JOINED=$(IFS=:; echo "${ROI_AGG_JOB_IDS[*]}")
  ALLROI_DEP_ARGS=(--dependency="afterok:${DEP_JOINED}")
fi
ROIS_JOINED=$(IFS=:; echo "${SELECTED_ROIS[*]}")
echo "Submitting final cross-ROI aggregate for ${#SELECTED_ROIS[@]} ROIs"
RAW=$(submit_cmd sbatch --parsable "${ALLROI_DEP_ARGS[@]}" \
  --export="ALL,VARIANT=${VARIANT},STUDY_ROOT=${STUDY_ROOT},ALLROI_ROOT=${ALLROI_ROOT},ROIS_EXPORT=${ROIS_JOINED}" \
  -p hugo --account geoffrey_hugo \
  jobs/supervised_correction_joint_spatial_qa_all_rois_aggregate.job)
ALLROI_JOB_ID="$(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
echo "Final all-ROI aggregate job: ${ALLROI_JOB_ID}"
echo "Expected ZIP: ${STUDY_ROOT}/joint_spatial_qa_all_rois/${VARIANT}_joint_spatial_all_roi_results_no_models.zip"
