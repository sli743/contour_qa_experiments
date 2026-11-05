#!/usr/bin/env bash
set -euo pipefail

# Diagnostic run of the trained Lung_L month-temporal ablation models on the
# Month1+Month2 development (training1/validation1) real autosegmentations.
# Not for reporting: it shows how the models treat inputs they were fit on,
# e.g. whether NSR-QA removes contoured stomach/large-bowel gas.
#
# Tasks come from ${STUDY_ROOT}/experiment_manifest.csv:
#   0 = nsrqa_corruption, 1 = no_corruption (identity baseline)
#   ARRAY=0 bash scripts/dino_shape/submit_lung_l_nsrqa_month_temporal_dev_reconstruction.sh
#   SKIP_PREPARE=1 ...   reuse an existing ${STUDY_ROOT}/actual_autoseg_preprocessed_dev

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_corruption_ablation_month_temporal_v1}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
MODE="${MODE:-autoseg_centered}"
ARRAY="${ARRAY:-0,1}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

cd "${PROJECT_ROOT}"
for path in \
  jobs/lung_l_nsrqa_month_temporal_dev_reconstruction_prepare.job \
  jobs/lung_l_nsrqa_month_temporal_dev_reconstruction_eval.job \
  scripts/dino_shape/run_lung_l_nsrqa_month_temporal_dev_reconstruction_task.sh; do
  [[ -f "${path}" ]] || { echo "ERROR: missing ${PROJECT_ROOT}/${path}" >&2; exit 2; }
done
[[ -s "${STUDY_ROOT}/experiment_manifest.csv" ]] || { echo "ERROR: missing ${STUDY_ROOT}/experiment_manifest.csv" >&2; exit 2; }

mkdir -p "${STUDY_ROOT}/logs"
EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},RAW_DATA_ROOT=${RAW_DATA_ROOT},MODE=${MODE}"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

DEPENDENCY=()
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  PREP_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
    --export="${EXPORTS}" \
    --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
    --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
    jobs/lung_l_nsrqa_month_temporal_dev_reconstruction_prepare.job)"
  PREP="$(jobid "${PREP_RAW}")"
  echo "Submitted dev autoseg preprocessing: ${PREP}"
  DEPENDENCY=(--dependency="afterok:${PREP}")
fi

EVAL_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="${ARRAY}" \
  jobs/lung_l_nsrqa_month_temporal_dev_reconstruction_eval.job)"
EVAL="$(jobid "${EVAL_RAW}")"
echo "Submitted dev reconstruction (array ${ARRAY}): ${EVAL}"

cat <<EOF2

Lung_L month-temporal development reconstruction submitted (diagnostic only)
  prepare:  ${PREP:-skipped}
  evaluate: ${EVAL}

Inputs (training1 + validation1, ${MODE}):
  ${STUDY_ROOT}/actual_autoseg_preprocessed_dev
Outputs per arm:
  ${STUDY_ROOT}/evaluations_dev/<arm>/seed<seed>/actual_autoseg_dev.csv
  ${STUDY_ROOT}/evaluations_dev/<arm>/seed<seed>/actual_autoseg_cases/${MODE}/Lung_L/<split>/patient_<id>/
Logs:
  ${STUDY_ROOT}/logs
EOF2
