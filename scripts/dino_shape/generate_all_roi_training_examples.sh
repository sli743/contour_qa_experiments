#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/dino_shape/all_roi_nsrqa_registry.yaml}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
CONDA_ENV="${CONDA_ENV:-3DINO}"
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
[[ -s "${MANIFEST}" ]] || python scripts/dino_shape/prepare_all_roi_nsrqa_study.py --registry "${REGISTRY}" --study-root "${STUDY_ROOT}"
mkdir -p "${STUDY_ROOT}/training_corruption_examples"
tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r roi_index registry_index roi data_group profile global_thr local_thr pilot_cfg bench_cfg random_cfg nsrqa_cfg; do
  # Backward-compatible with manifests written by the earlier preflight,
  # whose csv writer used CRLF line endings. The carriage return otherwise
  # becomes part of the final config path (for example, nsrqa.yaml\r).
  nsrqa_cfg="${nsrqa_cfg%$'\r'}"
  echo "=== ${roi}: standardized NSR-QA corruption examples (${profile}) ==="
  rm -rf "${STUDY_ROOT}/training_corruption_examples/${roi}"
  python -m evaluation.visualization.export_one_patient_training_corruptions \
    --config "${nsrqa_cfg}" --split validation --require-ct \
    --output-dir "${STUDY_ROOT}/training_corruption_examples/${roi}"
done

echo "All standardized examples: ${STUDY_ROOT}/training_corruption_examples"
