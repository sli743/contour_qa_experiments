#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
ROI="${1:-Lung_L}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SPLIT="${SPLIT:-test}"
case "${ROI}" in
  Lung_L) CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Lung_L_v1_eval.yaml}"; THRESH=20 ;;
  Bronchus) CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_stochastic_region_dropout_Bronchus_v1_eval.yaml}"; THRESH=10 ;;
  *) echo "Usage: $0 {Lung_L|Bronchus}" >&2; exit 2 ;;
esac
cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
EXP_ROOT=$(python - <<PY
import yaml
c=yaml.safe_load(open("${CONFIG}")); print(c["evaluation"]["output_root"])
PY
)
python -m evaluation.probes.eval_dino_shape_stochastic_region_dropout_probe_sweep --config "${CONFIG}" --split "${SPLIT}" --out_subdir "probe_eval_sweep_${SPLIT}" --inference_batch_size "${BATCH_SIZE}"
python -m evaluation.probes.export_dino_shape_denoising_probe_contours --config "${CONFIG}" --split "${SPLIT}" --output_subdir "probe_visualization_${SPLIT}" --inference_batch_size "${BATCH_SIZE}" --layout flat --save_metrics
python -m analysis.report_probe_model_performance --csv "${EXP_ROOT}/probe_eval_sweep_${SPLIT}/probe_eval_sweep_detailed.csv" --roi "${ROI}" --volume-threshold-pct "${THRESH}" --exclude-shifts --output-dir "${EXP_ROOT}/performance_report_${SPLIT}"
