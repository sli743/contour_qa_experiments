#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
COMPARE_ROOT="${COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
TEMPORAL_SPLIT_YAML="${TEMPORAL_SPLIT_YAML:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
FORCE_DIAGNOSTICS="${FORCE_DIAGNOSTICS:-0}"
SKIP_DEEP_PROBE_GEOMETRY="${SKIP_DEEP_PROBE_GEOMETRY:-0}"
BOUNDARY_MARGIN_MM="${BOUNDARY_MARGIN_MM:-4.5}"
SHORT_PERCENTILE="${SHORT_PERCENTILE:-10}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"

[[ -s "${COMPARE_ROOT}/manifests/model_configs.tsv" ]] || {
  echo "Missing completed comparison study/model manifest: ${COMPARE_ROOT}/manifests/model_configs.tsv" >&2; exit 2;
}
mkdir -p "${COMPARE_ROOT}/logs"

export_spec="ALL,PROJECT_ROOT=${PROJECT_ROOT},COMPARE_ROOT=${COMPARE_ROOT},RAW_DATA_ROOT=${RAW_DATA_ROOT},TEMPORAL_SPLIT_YAML=${TEMPORAL_SPLIT_YAML},FORCE_DIAGNOSTICS=${FORCE_DIAGNOSTICS},SKIP_DEEP_PROBE_GEOMETRY=${SKIP_DEEP_PROBE_GEOMETRY},BOUNDARY_MARGIN_MM=${BOUNDARY_MARGIN_MM},SHORT_PERCENTILE=${SHORT_PERCENTILE}"
cd "${PROJECT_ROOT}"
job_raw="$(sbatch --parsable \
  -p hugo \
  --account=geoffrey_hugo \
  --time="${TIME_LIMIT}" \
  --job-name=spcord_nsrqa_diag \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --output="${COMPARE_ROOT}/logs/slurm-%x-%j.out" \
  --error="${COMPARE_ROOT}/logs/slurm-%x-%j.err" \
  --export="${export_spec}" \
  jobs/spinalcord_nsrqa_diagnostics.job)"
job_id="${job_raw%%;*}"
cat <<EOF
Submitted SpinalCord four-level diagnostics: ${job_id}

Outputs:
  ${COMPARE_ROOT}/diagnostics/four_level_report/report.md
  ${COMPARE_ROOT}/diagnostics/acquisition_audit/incomplete_acquisition_patient_list.csv
  ${COMPARE_ROOT}/clean_approved_reconstruction/

Log:
  ${COMPARE_ROOT}/logs/slurm-spcord_nsrqa_diag-${job_id}.out
EOF
