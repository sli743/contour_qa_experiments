#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
python -m evaluation.probes.eval_dino_shape_probe_sweep_assd \
  --config "$CONFIG" \
  --split test \
  --out_subdir probe_eval_sweep \
  --threshold 0.5 \
  --sample_ratio 1.0 \
  --seed 88 \
  --save_vis \
  --vis_max_cases 3 \
  --drop_inf_in_plots
