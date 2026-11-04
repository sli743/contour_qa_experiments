#!/usr/bin/env bash
# Run data-vs-capacity diagnostics sequentially on the GPU in the current interactive allocation.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/supervised_autoseg_correction}"
ROI="${1:-Bronchus}"
FOLD="${2:-0}"
cd "${PROJECT_ROOT}"
source scripts/dino_shape/activate_3dino_env.sh
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python scripts/dino_shape/prepare_supervised_correction_diagnostics.py --study-root "${STUDY_ROOT}" --roi "${ROI}" --fold "${FOLD}"
DROOT="${STUDY_ROOT}/diagnostics/${ROI}/fold_${FOLD}"
TASKS="${DROOT}/diagnostic_tasks.tsv"
while IFS=$'\t' read -r idx experiment kind fraction augmentation train_n validation_n config; do
  [[ "${idx}" == "task_index" ]] && continue
  echo "============================================================"
  echo "Running ${experiment}: kind=${kind} train_n=${train_n} val_n=${validation_n} augmentation=${augmentation}"
  echo "Config: ${config}"
  python -m training.train_supervised_autoseg_correction_decoder --config "${config}" --fresh
  python -m evaluation.supervised_correction.eval_supervised_correction_diagnostic \
    --config "${config}" \
    --output-dir "${DROOT}/evaluation/${experiment}"
done < "${TASKS}"
python -m analysis.summarize_supervised_correction_diagnostics --diagnostic-root "${DROOT}"
echo "Done. Summary: ${DROOT}/diagnostic_summary.csv"
