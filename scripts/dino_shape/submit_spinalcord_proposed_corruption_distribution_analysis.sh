#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
LOG_DIR="${STUDY_ROOT}/logs"
mkdir -p "${LOG_DIR}"
export PROJECT_ROOT STUDY_ROOT
sbatch \
  -p hugo \
  --account=geoffrey_hugo \
  --output="${LOG_DIR}/slurm-spcord-proposed-dist-%j.out" \
  --error="${LOG_DIR}/slurm-spcord-proposed-dist-%j.out" \
  "${PROJECT_ROOT}/jobs/spinalcord_proposed_corruption_distribution_analysis.job" \
  "$@"
