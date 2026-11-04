#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${STUDY_ROOT}/spinalcord_corruption_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIBRATION_ROOT}/proposed_distribution_validation_v4}"
PROPOSAL_CONFIG="${PROPOSAL_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_proposed_corruption_distribution_v4.yaml}"
LOG_DIR="${STUDY_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export PROJECT_ROOT STUDY_ROOT CALIBRATION_ROOT OUTPUT_DIR PROPOSAL_CONFIG
sbatch \
  -p hugo \
  --account=geoffrey_hugo \
  --output="${LOG_DIR}/slurm-spcord-proposed-v4-%j.out" \
  --error="${LOG_DIR}/slurm-spcord-proposed-v4-%j.out" \
  "${PROJECT_ROOT}/jobs/spinalcord_proposed_corruption_distribution_v4_analysis.job" \
  "$@"
