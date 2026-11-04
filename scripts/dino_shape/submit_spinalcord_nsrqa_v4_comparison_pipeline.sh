#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
SOURCE_STUDY_ROOT="${SOURCE_STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
COMPARE_ROOT="${COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
V4_CONFIG="${V4_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_proposed_corruption_distribution_v4.yaml}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-${RAW_DATA_ROOT}/metrics.csv}"
DOMAIN_CSV="${DOMAIN_CSV:-/home/shaohan/radformation-data/OOD_Domain_Assignment_MAD_TOTAL.csv}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-bbox_midpoint}"
SEED="${SEED:-88}"
N_BOOT="${N_BOOT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FRESH_TRAINING="${FRESH_TRAINING:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"

[[ -s "${SOURCE_STUDY_ROOT}/manifests/temporal_split_summary.json" ]] || {
  echo "Missing prepared month-temporal study: ${SOURCE_STUDY_ROOT}" >&2; exit 2;
}
[[ -s "${V4_CONFIG}" ]] || { echo "Missing v4 config: ${V4_CONFIG}" >&2; exit 2; }
[[ -s "${CLINICAL_METRICS_CSV}" ]] || { echo "Missing clinical metrics: ${CLINICAL_METRICS_CSV}" >&2; exit 2; }
mkdir -p "${COMPARE_ROOT}/logs"

export_spec="ALL,PROJECT_ROOT=${PROJECT_ROOT},SOURCE_STUDY_ROOT=${SOURCE_STUDY_ROOT},COMPARE_ROOT=${COMPARE_ROOT},V4_CONFIG=${V4_CONFIG},RAW_DATA_ROOT=${RAW_DATA_ROOT},CLINICAL_METRICS_CSV=${CLINICAL_METRICS_CSV},DOMAIN_CSV=${DOMAIN_CSV},CLINICAL_METRICS_DUPLICATE_POLICY=${CLINICAL_METRICS_DUPLICATE_POLICY},AUTOSEG_CENTER_METHOD=${AUTOSEG_CENTER_METHOD},SEED=${SEED},N_BOOT=${N_BOOT},BATCH_SIZE=${BATCH_SIZE},FRESH_TRAINING=${FRESH_TRAINING},FORCE_EVAL=${FORCE_EVAL}"
cd "${PROJECT_ROOT}"
job_raw="$(sbatch --parsable \
  -p hugo \
  --account=geoffrey_hugo \
  --time="${TIME_LIMIT}" \
  --job-name=spcord_nsrqa_v4cmp \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --output="${COMPARE_ROOT}/logs/slurm-%x-%j.out" \
  --error="${COMPARE_ROOT}/logs/slurm-%x-%j.err" \
  --export="${export_spec}" \
  jobs/spinalcord_nsrqa_v4_comparison_pipeline.job)"
job_id="${job_raw%%;*}"
cat <<EOF
Submitted SpinalCord previous-generation vs v4 NSR-QA comparison: ${job_id}

Scientific contrast:
  previous-generation generic stochastic corruption
  versus
  v4 development-calibrated SpinalCord corruption

Primary endpoint:
  actual Month3 autoseg QA performance (P75 AUROC/AUPRC + Spearman for 1-Dice/HD95)

Study root:
  ${COMPARE_ROOT}
Log:
  ${COMPARE_ROOT}/logs/slurm-spcord_nsrqa_v4cmp-${job_id}.out
EOF
