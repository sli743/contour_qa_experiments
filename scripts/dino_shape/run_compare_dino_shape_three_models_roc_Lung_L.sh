#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
N_BOOT="${N_BOOT:-2000}"

PILOT_CONFIG="${PILOT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml}"
RANDOM_CONFIG="${RANDOM_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Lung_L_v1_eval.yaml}"

PILOT_CSV="${PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep/probe_eval_sweep_detailed.csv}"
BENCHMARK_CSV="${BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
RANDOM_CSV="${RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"

OUT_DIR="${OUT_DIR:-/scratch/shaohan/contour_QA/roc_comparison/dino_shape_three_models_Lung_L}"
PAIR1="${OUT_DIR}/pilot_vs_random_morphology"
PAIR2="${OUT_DIR}/benchmark_upper_limit_vs_random_morphology"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m analysis.roc.compare_dino_shape_roc_reports \
  --pilot_csv "${PILOT_CSV}" --denoising_csv "${RANDOM_CSV}" \
  --pilot_config "${PILOT_CONFIG}" --denoising_config "${RANDOM_CONFIG}" \
  --out_dir "${PAIR1}" --roi Lung_L --split test --volume_threshold_pct 20 \
  --clean_policy one_per_patient --n_boot "${N_BOOT}" --seed 88

python -m analysis.roc.compare_dino_shape_roc_reports \
  --pilot_csv "${BENCHMARK_CSV}" --denoising_csv "${RANDOM_CSV}" \
  --pilot_config "${BENCHMARK_CONFIG}" --denoising_config "${RANDOM_CONFIG}" \
  --out_dir "${PAIR2}" --roi Lung_L --split test --volume_threshold_pct 20 \
  --clean_policy one_per_patient --n_boot "${N_BOOT}" --seed 88

python -m analysis.roc.summarize_three_dino_shape_models \
  --pilot_random_dir "${PAIR1}" \
  --benchmark_random_dir "${PAIR2}" \
  --out_dir "${OUT_DIR}"

if command -v pandoc >/dev/null 2>&1; then
  find "${OUT_DIR}" -type f -name '*.md' -print0 | while IFS= read -r -d '' md; do
    pandoc "${md}" --standalone --toc --embed-resources --highlight-style=tango \
      --metadata title="$(basename "${md%.md}")" -o "${md%.md}.html"
  done
else
  echo "Pandoc not found; Markdown reports were still generated."
fi

echo "Main report: ${OUT_DIR}/report/three_model_ROC_comparison.md"
echo "HTML report: ${OUT_DIR}/report/three_model_ROC_comparison.html"
