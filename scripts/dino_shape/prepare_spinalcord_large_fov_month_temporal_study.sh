#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_month_temporal_NSRQA_study}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
FRESH_DATA="${FRESH_DATA:-1}"

# Reuse the coverage-safe large-FOV preparer. Supplying SPLIT_CONFIG switches
# patient membership to the established temporal policy:
#   Month1+Month2 -> development (stratified train/validation)
#   Month3        -> held-out testing1
# while preserving bbox-midpoint centering and development-only FOV sizing.
export PROJECT_ROOT STUDY_ROOT SPLIT_CONFIG FRESH_DATA
exec bash "${PROJECT_ROOT}/scripts/dino_shape/prepare_spinalcord_large_fov_nsrqa_study.sh"
