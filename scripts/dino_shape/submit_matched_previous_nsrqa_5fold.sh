#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ALLROI_ROOT="${ALLROI_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
BASELINE_ROOT="${BASELINE_ROOT:-${STUDY_ROOT}/matched_previous_nsrqa_5fold}"
ROI="${1:?ROI required}"
VARIANT="${VARIANT:-large_decoder_ft_last2}"
FOLDS="${FOLDS:-0,1,2,3,4}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
QA_N_BOOT="${QA_N_BOOT:-5000}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_ROOT}"
IFS=',' read -r -a FOLD_ARRAY <<< "${FOLDS}"
EVAL_JOB_IDS=()

submit_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'; printf ' %q' "$@"; printf '\n'; echo DRYRUN
  else
    "$@"
  fi
}

for FOLD_RAW in "${FOLD_ARRAY[@]}"; do
  FOLD="$(echo "${FOLD_RAW}" | xargs)"
  MANIFEST="${STUDY_ROOT}/manifests/${ROI}/fold_${FOLD}.csv"
  REF_CFG="${ALLROI_ROOT}/configs/${ROI}/nsrqa.yaml"
  [[ -f "${MANIFEST}" ]] || { echo "Missing current fold manifest: ${MANIFEST}" >&2; exit 2; }
  [[ -f "${REF_CFG}" ]] || { echo "Missing previous NSR-QA reference config: ${REF_CFG}" >&2; exit 2; }

  FOLD_ROOT="${BASELINE_ROOT}/${ROI}/fold_${FOLD}"
  CHECKPOINT="${FOLD_ROOT}/models/matched_previous_nsrqa_${ROI}_fold${FOLD}/checkpoints/best_decoder.pth"
  SENTINEL="${FOLD_ROOT}/analysis/nsrqa_regression_summary.csv"
  TRAIN_JOB=""
  if [[ "${FORCE_TRAIN}" == "1" || ! -f "${CHECKPOINT}" ]]; then
    RAW=$(submit_cmd sbatch --parsable \
      --export="ALL,ROI=${ROI},FOLD=${FOLD},STUDY_ROOT=${STUDY_ROOT},ALLROI_ROOT=${ALLROI_ROOT},BASELINE_ROOT=${BASELINE_ROOT}" \
      -p hugo --account geoffrey_hugo jobs/matched_previous_nsrqa_train.job)
    TRAIN_JOB="$(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
    echo "NSR-QA train ROI=${ROI} fold=${FOLD}: ${TRAIN_JOB}"
  else
    echo "Reusing NSR-QA checkpoint: ${CHECKPOINT}"
  fi

  if [[ -z "${TRAIN_JOB}" && "${FORCE_EVAL}" != "1" && -f "${SENTINEL}" ]]; then
    echo "Reusing matched NSR-QA evaluation: ${SENTINEL}"
    continue
  fi
  DEP=()
  if [[ -n "${TRAIN_JOB}" && "${TRAIN_JOB}" != "DRYRUN" ]]; then DEP=(--dependency="afterok:${TRAIN_JOB}"); fi
  RAW=$(submit_cmd sbatch --parsable "${DEP[@]}" \
    --export="ALL,ROI=${ROI},FOLD=${FOLD},STUDY_ROOT=${STUDY_ROOT},ALLROI_ROOT=${ALLROI_ROOT},BASELINE_ROOT=${BASELINE_ROOT}" \
    -p hugo --account geoffrey_hugo jobs/matched_previous_nsrqa_eval.job)
  J="$(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
  [[ "${J}" == "DRYRUN" ]] || EVAL_JOB_IDS+=("${J}")
  echo "NSR-QA eval ROI=${ROI} fold=${FOLD}: ${J}"
done

# Comparison requires the current joint-spatial result for all five folds. Fail
# explicitly if it is absent rather than silently comparing against partial data.
CURRENT_OK=1
for FOLD_RAW in "${FOLD_ARRAY[@]}"; do
  FOLD="$(echo "${FOLD_RAW}" | xargs)"
  P="${STUDY_ROOT}/capacity_finetune_ablation/${ROI}/fold_${FOLD}/${VARIANT}/spatial_qa_head_analysis/spatial_qa_regression_summary.csv"
  [[ -f "${P}" ]] || CURRENT_OK=0
done
if [[ "${CURRENT_OK}" == "0" && "${DRY_RUN}" != "1" ]]; then
  echo "Matched NSR-QA jobs submitted, but current joint-spatial 5-fold outputs are incomplete." >&2
  echo "After the current-method sweep finishes, run:" >&2
  echo "  bash scripts/dino_shape/submit_current_vs_previous_nsrqa_comparison.sh ${ROI}" >&2
  exit 0
fi

DEP=()
if (( ${#EVAL_JOB_IDS[@]} > 0 )); then D=$(IFS=:; echo "${EVAL_JOB_IDS[*]}"); DEP=(--dependency="afterok:${D}"); fi
RAW=$(submit_cmd sbatch --parsable "${DEP[@]}" \
  --export="ALL,ROI=${ROI},VARIANT=${VARIANT},STUDY_ROOT=${STUDY_ROOT},QA_N_BOOT=${QA_N_BOOT}" \
  -p hugo --account geoffrey_hugo jobs/current_vs_previous_nsrqa_compare.job)
echo "Comparison job: $(echo "${RAW}" | tail -n1 | cut -d';' -f1)"
