#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
RESULT_DIR="${RESULT_DIR:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
echo "Probe results are expected at: $RESULT_DIR"
echo "Run the existing AUC/FDVC scripts after confirming their CLI arguments against this CSV:"
echo "  $RESULT_DIR/probe_eval_sweep_detailed.csv"
