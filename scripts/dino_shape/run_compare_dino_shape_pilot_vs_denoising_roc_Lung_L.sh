#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONDA_ENV="${CONDA_ENV:-3DINO}"

PILOT_CONFIG="${PILOT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml}"
DENOISING_CONFIG="${DENOISING_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml}"

PILOT_CSV="${PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep/probe_eval_sweep_detailed.csv}"
DENOISING_CSV="${DENOISING_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
OUT_DIR="${OUT_DIR:-/scratch/shaohan/contour_QA/roc_comparison/dino_shape_pilot_vs_denoising_Lung_L}"
N_BOOT="${N_BOOT:-2000}"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m analysis.roc.compare_dino_shape_roc_reports \
  --pilot_csv "${PILOT_CSV}" \
  --denoising_csv "${DENOISING_CSV}" \
  --pilot_config "${PILOT_CONFIG}" \
  --denoising_config "${DENOISING_CONFIG}" \
  --out_dir "${OUT_DIR}" \
  --roi Lung_L \
  --split test \
  --volume_threshold_pct 20 \
  --clean_policy one_per_patient \
  --n_boot "${N_BOOT}" \
  --seed 88 \
  "$@"

REPORT_DIR="${OUT_DIR}/report"

if command -v pandoc >/dev/null 2>&1; then
    echo "Converting Markdown reports to HTML..."

    for md in "${REPORT_DIR}"/*.md; do
        [ -f "$md" ] || continue

        html="${md%.md}.html"

        pandoc \
            "$md" \
            --standalone \
            --toc \
            --metadata title="$(basename "${md%.md}")" \
            -o "$html"
    done

    echo "HTML reports written to:"
    ls "${REPORT_DIR}"/*.html
else
    echo "Pandoc not found; skipping HTML conversion."
fi