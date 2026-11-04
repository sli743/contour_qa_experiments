#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
BANK_ROOT="${BANK_ROOT:-${STUDY_ROOT}/shared_probe_banks}"
CLINICAL_METRICS_CSV="${CLINICAL_METRICS_CSV:-${PROJECT_ROOT}/../radformation-data/data/metrics.csv}"
CLINICAL_METRICS_DUPLICATE_POLICY="${CLINICAL_METRICS_DUPLICATE_POLICY:-exclude}"
CLINICAL_SITE="${CLINICAL_SITE:-thorax}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-bbox_midpoint}"
SEED="${SEED:-88}"
N_BOOT="${N_BOOT:-5000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PREPARE_STUDY="${PREPARE_STUDY:-0}"
FRESH_DATA="${FRESH_DATA:-1}"
REUSE_COMPLETED="${REUSE_COMPLETED:-1}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"

cd "${PROJECT_ROOT}"
[[ -f jobs/spinalcord_large_fov_month_temporal_pipeline.job ]] || { echo "Missing pipeline job file" >&2; exit 2; }
[[ -f scripts/dino_shape/run_spinalcord_large_fov_month_temporal_pipeline.sh ]] || { echo "Missing pipeline runner" >&2; exit 2; }
[[ -s "${CLINICAL_METRICS_CSV}" ]] || { echo "Missing clinical metrics CSV: ${CLINICAL_METRICS_CSV}" >&2; exit 2; }

TEMPORAL_META="${STUDY_ROOT}/manifests/temporal_split_summary.json"
if [[ "${PREPARE_STUDY}" != "1" && ! -s "${TEMPORAL_META}" ]]; then
  cat >&2 <<EOF2
Temporal study is not prepared:
  ${TEMPORAL_META}

Prepare and inspect it first:
  FRESH_DATA=1 bash scripts/dino_shape/prepare_spinalcord_large_fov_month_temporal_study.sh

Or set PREPARE_STUDY=1 to perform preparation as Stage 0 inside the same Slurm allocation.
EOF2
  exit 2
fi

mkdir -p "${STUDY_ROOT}/logs"

export_spec="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},REG_ROOT=${REG_ROOT},BANK_ROOT=${BANK_ROOT},CLINICAL_METRICS_CSV=${CLINICAL_METRICS_CSV},CLINICAL_METRICS_DUPLICATE_POLICY=${CLINICAL_METRICS_DUPLICATE_POLICY},CLINICAL_SITE=${CLINICAL_SITE},AUTOSEG_CENTER_METHOD=${AUTOSEG_CENTER_METHOD},SEED=${SEED},N_BOOT=${N_BOOT},BATCH_SIZE=${BATCH_SIZE},PREPARE_STUDY=${PREPARE_STUDY},FRESH_DATA=${FRESH_DATA},REUSE_COMPLETED=${REUSE_COMPLETED}"

job_raw="$(sbatch --parsable \
  -p hugo \
  --account=geoffrey_hugo \
  --time="${TIME_LIMIT}" \
  --job-name=spcord_mt_pipeline \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  --export="${export_spec}" \
  jobs/spinalcord_large_fov_month_temporal_pipeline.job)"
job_id="${job_raw%%;*}"

cat <<EOF2
Submitted one SpinalCord month-temporal large-FOV pipeline job: ${job_id}

This is one Slurm allocation; it does not submit child jobs.
Stages run sequentially:
  1. pilot / benchmark / random_morphology / NSRQA training
  2. shared synthetic-probe evaluation + four-model report
  3. actual-autoseg reconstruction + regression
  4. P75/P50 ranking/classification
  5. ranking/classification report

Study root:
  ${STUDY_ROOT}
Log:
  ${STUDY_ROOT}/logs/slurm-spcord_mt_pipeline-${job_id}.out

REUSE_COMPLETED=${REUSE_COMPLETED}: on resubmission, only stages with both a success marker and expected outputs are skipped.
EOF2
