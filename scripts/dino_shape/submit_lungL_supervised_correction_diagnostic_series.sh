#!/usr/bin/env bash
set -euo pipefail

# Run the same v5/v6 supervised autoseg->approved diagnostic series used for
# Bronchus, but for Lung_L. Existing trained checkpoints are reused by default.
# Missing checkpoints trigger the corresponding training/evaluation job first.
#
# Usage:
#   bash scripts/dino_shape/submit_lungL_supervised_correction_diagnostic_series.sh [fold] [split] [local_radius] [action_weight]
#
# Example (same settings as Bronchus diagnostic):
#   bash scripts/dino_shape/submit_lungL_supervised_correction_diagnostic_series.sh 0 validation 6 4.0
#
# Optional environment variables:
#   FORCE_TRAIN=1   retrain v5 and v6 even if best_decoder.pth already exists
#   PROJECT_ROOT=... STUDY_ROOT=...

ROI="Lung_L"
FOLD="${1:-0}"
SPLIT="${2:-validation}"
LOCAL_RADIUS="${3:-6}"
ACTION_WEIGHT="${4:-4.0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"

cd "${PROJECT_ROOT}"

BASE_CONFIG="${STUDY_ROOT}/configs/${ROI}/fold_${FOLD}.yaml"
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Missing base supervised-correction config: ${BASE_CONFIG}" >&2
  echo "Run the all-ROI supervised-correction preprocessing/prepare stage first." >&2
  exit 1
fi

submit_training_if_needed() {
  local version="$1"
  local model_dir="$2"
  local job_file="$3"
  local ckpt="${STUDY_ROOT}/${model_dir}/${ROI}/fold_${FOLD}/models/${version}/checkpoints/best_decoder.pth"

  # The generated config controls experiment_name, so checkpoint location may
  # differ slightly by version. Prefer a filesystem search under this fold.
  local fold_root="${STUDY_ROOT}/${model_dir}/${ROI}/fold_${FOLD}"
  local existing_ckpt=""
  if [[ -d "${fold_root}" ]]; then
    existing_ckpt=$(find "${fold_root}" -type f -path '*/checkpoints/best_decoder.pth' -print -quit 2>/dev/null || true)
  fi

  if [[ "${FORCE_TRAIN}" != "1" && -n "${existing_ckpt}" ]]; then
    echo "Reusing ${version} checkpoint: ${existing_ckpt}" >&2
    printf '%s' ""
    return 0
  fi

  local jid
  if [[ "${version}" == "v6" ]]; then
    jid=$(sbatch \
      -p hugo \
      --account geoffrey_hugo \
      --export="ALL,ROI=${ROI},FOLD=${FOLD},ACTION_WEIGHT=${ACTION_WEIGHT}" \
      --parsable \
      "${job_file}")
  else
    jid=$(sbatch \
      -p hugo \
      --account geoffrey_hugo \
      --export="ALL,ROI=${ROI},FOLD=${FOLD}" \
      --parsable \
      "${job_file}")
  fi
  echo "Submitted ${ROI} ${version} training/evaluation job: ${jid}" >&2
  printf '%s' "${jid}"
}

submit_score_diag() {
  local version="$1"
  local dependency="$2"
  local dep_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args=(--dependency="afterok:${dependency}")
  fi

  sbatch \
    -p hugo \
    --account geoffrey_hugo \
    "${dep_args[@]}" \
    --export="ALL,ROI=${ROI},FOLD=${FOLD},MODEL_VERSION=${version},SPLIT=${SPLIT},LOCAL_RADIUS=${LOCAL_RADIUS}" \
    --parsable \
    jobs/supervised_correction_action_score_diagnostic.job
}

submit_gate_diag() {
  local dependency="$1"
  local dep_args=()
  if [[ -n "${dependency}" ]]; then
    dep_args=(--dependency="afterok:${dependency}")
  fi

  sbatch \
    -p hugo \
    --account geoffrey_hugo \
    "${dep_args[@]}" \
    --export="ALL,ROI=${ROI},FOLD=${FOLD},MODEL_VERSION=v6,SPLIT=${SPLIT},LOCAL_RADIUS=${LOCAL_RADIUS}" \
    --parsable \
    jobs/supervised_correction_patient_gating_diagnostic.job
}

V5_TRAIN=$(submit_training_if_needed \
  v5 threeclass_correction_v5 jobs/supervised_correction_threeclass_v5.job)
V6_TRAIN=$(submit_training_if_needed \
  v6 threeclass_correction_v6_weighted_ce jobs/supervised_correction_threeclass_v6.job)

V5_SCORE=$(submit_score_diag v5 "${V5_TRAIN}")
echo "Submitted ${ROI} v5 action-score diagnostic: ${V5_SCORE}"

V6_SCORE=$(submit_score_diag v6 "${V6_TRAIN}")
echo "Submitted ${ROI} v6 action-score diagnostic: ${V6_SCORE}"

# The gating job reruns exact v6 score inference and then performs patient-level
# analysis. Depend on v6 score diagnostic too so the sequence is easy to audit
# and the first diagnostic outputs are guaranteed to exist before gating runs.
V6_GATE=$(submit_gate_diag "${V6_SCORE}")
echo "Submitted ${ROI} v6 patient-gating diagnostic: ${V6_GATE}"

echo
echo "Lung_L diagnostic series submitted."
echo "  ROI             : ${ROI}"
echo "  fold            : ${FOLD}"
echo "  split           : ${SPLIT}"
echo "  local radius    : ${LOCAL_RADIUS} vox"
echo "  v6 action weight: ${ACTION_WEIGHT}"
echo "  v5 train        : ${V5_TRAIN:-reused checkpoint}"
echo "  v6 train        : ${V6_TRAIN:-reused checkpoint}"
echo "  v5 score diag   : ${V5_SCORE}"
echo "  v6 score diag   : ${V6_SCORE}"
echo "  v6 gating diag  : ${V6_GATE}"
echo
echo "Expected outputs:"
echo "  ${STUDY_ROOT}/action_score_diagnostics/threeclass_correction_v5/${ROI}/fold_${FOLD}/${SPLIT}"
echo "  ${STUDY_ROOT}/action_score_diagnostics/threeclass_correction_v6_weighted_ce/${ROI}/fold_${FOLD}/${SPLIT}"
