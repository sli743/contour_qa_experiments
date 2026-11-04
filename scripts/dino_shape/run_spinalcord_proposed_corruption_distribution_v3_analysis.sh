#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${STUDY_ROOT}/spinalcord_corruption_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-${CALIBRATION_ROOT}/proposed_distribution_validation_v3}"
PROPOSAL_CONFIG="${PROPOSAL_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_proposed_corruption_distribution_v3.yaml}"
NSRQA_CONFIG="${NSRQA_CONFIG:-${STUDY_ROOT}/configs/SpinalCord/nsrqa.yaml}"
WORKERS="${WORKERS:-8}"
SAMPLES_PER_CASE="${SAMPLES_PER_CASE:-0}"

export PROJECT_ROOT STUDY_ROOT CALIBRATION_ROOT OUTPUT_DIR PROPOSAL_CONFIG NSRQA_CONFIG WORKERS SAMPLES_PER_CASE
exec bash "${PROJECT_ROOT}/scripts/dino_shape/run_spinalcord_proposed_corruption_distribution_analysis.sh" "$@"
