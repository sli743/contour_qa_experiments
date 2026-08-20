#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
TASK_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${TASK_INDEX}" ]] || { echo "Usage: $0 TASK_INDEX (or SLURM_ARRAY_TASK_ID)" >&2; exit 2; }
TASKS="${STUDY_ROOT}/manifests/training_tasks.tsv"
[[ -s "${TASKS}" ]] || { echo "Missing training task manifest: ${TASKS}. Run prepare_all_roi_nsrqa_study.py first." >&2; exit 2; }
line="$(awk -F '\t' -v idx="${TASK_INDEX}" 'NR>1 && $1==idx {print; exit}' "${TASKS}")"
[[ -n "${line}" ]] || { echo "No task ${TASK_INDEX} in ${TASKS}" >&2; exit 2; }
line="${line%$'\r'}"
IFS=$'\t' read -r task_index roi model config <<< "${line}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset PYTORCH_CUDA_ALLOC_CONF || true

echo "Training task ${task_index}: ROI=${roi} model=${model} config=${config}"
if [[ "${FRESH:-1}" == "1" ]]; then
  experiment_dir="$(python - "${config}" <<'PYCFG'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["output_root"]) / cfg["experiment_name"])
PYCFG
)"
  echo "Fresh all-ROI study run: removing ${experiment_dir}"
  rm -rf "${experiment_dir}"
fi
case "${model}" in
  pilot) python -m training.train_dino_shape_decoder --config "${config}" ;;
  benchmark) python -m training.train_dino_shape_denoising_decoder --config "${config}" ;;
  random_morphology) python -m training.train_dino_shape_random_morphology_decoder --config "${config}" ;;
  nsrqa) python -m training.train_dino_shape_stochastic_region_dropout_decoder --config "${config}" --fresh ;;
  *) echo "Unknown model: ${model}" >&2; exit 2 ;;
esac
