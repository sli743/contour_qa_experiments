#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Bronchus_probe_eval.yaml}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
SPLIT="${SPLIT:-test}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python -m evaluation.probes.eval_dino_shape_probe_sweep_assd   --config "${CONFIG}"   --split "${SPLIT}"   --out_subdir "probe_eval_sweep_${SPLIT}"   --threshold 0.5   --sample_ratio 1.0   --seed 88   --save_vis   --vis_max_cases 3   --drop_inf_in_plots   "$@"
