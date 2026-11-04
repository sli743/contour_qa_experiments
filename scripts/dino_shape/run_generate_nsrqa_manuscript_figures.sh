#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/shaohan/contour_QA/manuscript_figures/nsrqa}"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m analysis.manuscript.generate_nsrqa_manuscript_figures \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo "Manuscript figure directory: ${OUTPUT_DIR}"
