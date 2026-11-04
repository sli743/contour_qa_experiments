#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
idx="${1:?evaluation task index required}"
tasks="${STUDY_ROOT}/manifests/training_tasks.tsv"
config=$(awk -F '\t' -v i="$idx" 'NR>1 && $1==i {print $4; exit}' "$tasks")
[[ -n "$config" ]] || { echo "No evaluation task $idx" >&2; exit 2; }
python -m evaluation.supervised_correction.eval_supervised_autoseg_correction --config "$config" --study-root "$STUDY_ROOT"
