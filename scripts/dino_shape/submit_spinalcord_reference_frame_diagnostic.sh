#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
COMPARE_ROOT="${COMPARE_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_v4corr_compare}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
CENTER_METHOD="${CENTER_METHOD:-bbox_midpoint}"
FORCE_REFERENCE_FRAME="${FORCE_REFERENCE_FRAME:-0}"
MANUAL_INCOMPLETE_PATIENTS="${MANUAL_INCOMPLETE_PATIENTS:-95 129 145}"
TIME_LIMIT="${TIME_LIMIT:-06:00:00}"

[[ -s "${COMPARE_ROOT}/manifests/model_configs.tsv" ]] || {
  echo "Missing completed comparison study/model manifest: ${COMPARE_ROOT}/manifests/model_configs.tsv" >&2; exit 2;
}
mkdir -p "${COMPARE_ROOT}/logs"

export_spec="ALL,PROJECT_ROOT=${PROJECT_ROOT},COMPARE_ROOT=${COMPARE_ROOT},RAW_DATA_ROOT=${RAW_DATA_ROOT},CENTER_METHOD=${CENTER_METHOD},FORCE_REFERENCE_FRAME=${FORCE_REFERENCE_FRAME},MANUAL_INCOMPLETE_PATIENTS=${MANUAL_INCOMPLETE_PATIENTS}"
cd "${PROJECT_ROOT}"
job_raw="$(sbatch --parsable \
  -p hugo \
  --account=geoffrey_hugo \
  --time="${TIME_LIMIT}" \
  --job-name=spcord_ref_frame \
  --mail-type=ALL \
  --mail-user=shaohan@wustl.edu \
  --output="${COMPARE_ROOT}/logs/slurm-%x-%j.out" \
  --error="${COMPARE_ROOT}/logs/slurm-%x-%j.err" \
  --export="${export_spec}" \
  jobs/spinalcord_reference_frame_diagnostic.job)"
job_id="${job_raw%%;*}"
cat <<EOF
Submitted SpinalCord reference-frame diagnostic: ${job_id}

Report:
  ${COMPARE_ROOT}/diagnostics/reference_frame/report.md

Hybrid-mode viewer results are added under:
  ${COMPARE_ROOT}/actual_autoseg_results/v4_corruption/autoseg_xy_ct_z/
  ${COMPARE_ROOT}/clean_approved_reconstruction/v4_corruption/autoseg_xy_ct_z/

Log:
  ${COMPARE_ROOT}/logs/slurm-spcord_ref_frame-${job_id}.out
EOF
