#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
SEED="${SEED:-88}"
N_BOOT="${N_BOOT:-5000}"

cd "${PROJECT_ROOT}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rerun_roi_report() {
  local roi="$1"
  local global_threshold="$2"
  local local_depth="$3"
  local pilot_csv="$4"
  local bench_csv="$5"
  local random_csv="$6"
  local v3_csv="$7"
  local v3_vis="$8"
  local report_root="$9"
  local comparison_name="${10}"

  local pooled_out="${report_root}/pooled"
  local by_probe_out="${report_root}/by_probe"
  local master_report="${report_root}/report.md"

  for f in "${pilot_csv}" "${bench_csv}" "${random_csv}" "${v3_csv}"; do
    [[ -s "${f}" ]] || { echo "ERROR: missing evaluation CSV: ${f}" >&2; exit 1; }
  done
  [[ -d "${v3_vis}" ]] || { echo "ERROR: missing visualization directory: ${v3_vis}" >&2; exit 1; }

  echo
  echo "=== ${roi}: patient-cluster bootstrap (${N_BOOT} replicates) ==="
  rm -rf "${pooled_out}" "${by_probe_out}"
  mkdir -p "${pooled_out}" "${by_probe_out}"

  python -m analysis.generate_four_model_all_probes_two_criteria \
    --model "Pilot AE=${pilot_csv}" \
    --model "Benchmark-leakage DAE=${bench_csv}" \
    --model "Random-morphology DAE=${random_csv}" \
    --model "Stochastic-region-dropout DAE v3 symmetric=${v3_csv}" \
    --roi "${roi}" \
    --volume-threshold-pct "${global_threshold}" \
    --local-depth-threshold-mm "${local_depth}" \
    --exclude-shifts \
    --n-boot "${N_BOOT}" \
    --seed "${SEED}" \
    --output-dir "${pooled_out}"

  python -m analysis.generate_four_model_per_probe_reports \
    --model "Pilot AE=${pilot_csv}" \
    --model "Benchmark-leakage DAE=${bench_csv}" \
    --model "Random-morphology DAE=${random_csv}" \
    --model "Stochastic-region-dropout DAE v3 symmetric=${v3_csv}" \
    --roi "${roi}" \
    --volume-threshold-pct "${global_threshold}" \
    --local-depth-threshold-mm "${local_depth}" \
    --probe-visualization-dir "${v3_vis}" \
    --geometry-volume-tolerance-pct 0.5 \
    --exclude-shifts \
    --n-boot "${N_BOOT}" \
    --seed "${SEED}" \
    --output-dir "${by_probe_out}"

  python -m analysis.compose_four_model_shared_probe_report \
    --pooled-dir "${pooled_out}" \
    --by-probe-dir "${by_probe_out}" \
    --output "${master_report}" \
    --roi "${roi}" \
    --comparison-name "${comparison_name}" \
    --model-label "Pilot AE" \
    --model-label "Benchmark-leakage DAE" \
    --model-label "Random-morphology DAE" \
    --model-label "Stochastic-region-dropout DAE v3 symmetric"

  if command -v pandoc >/dev/null 2>&1; then
    (
      cd "${report_root}"
      pandoc report.md --standalone --embed-resources -o report.html || true
    )
  fi

  echo "Updated: ${master_report}"
}

LUNGL_PILOT="/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
LUNGL_BENCH="/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
LUNGL_RANDOM="/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
LUNGL_V3_EXP="/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Lung_L_v3_symmetric_changed_loss"
LUNGL_V3="${LUNGL_V3_EXP}/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
LUNGL_VIS="${LUNGL_V3_EXP}/probe_eval_shared_seed_test/probe_visualization_shared"
LUNGL_REPORT="/scratch/shaohan/contour_QA/roc_comparison/Lung_L_four_models_v3_symmetric_shared_probe"

BRONCHUS_PILOT="/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
BRONCHUS_BENCH="/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
BRONCHUS_RANDOM="/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
BRONCHUS_V3_EXP="/scratch/shaohan/contour_QA/dino_stochastic_region_dropout/dino_shape_stochastic_region_dropout_Bronchus_v3_symmetric_changed_loss"
BRONCHUS_V3="${BRONCHUS_V3_EXP}/probe_eval_shared_seed_test/probe_eval_sweep_detailed.csv"
BRONCHUS_VIS="${BRONCHUS_V3_EXP}/probe_eval_shared_seed_test/probe_visualization_shared"
BRONCHUS_REPORT="/scratch/shaohan/contour_QA/roc_comparison/Bronchus_four_models_v3_symmetric_shared_probe"

rerun_roi_report \
  "Lung_L" 20 10 \
  "${LUNGL_PILOT}" "${LUNGL_BENCH}" "${LUNGL_RANDOM}" "${LUNGL_V3}" \
  "${LUNGL_VIS}" "${LUNGL_REPORT}" \
  "stochastic v3 symmetric changed-region loss; patient-cluster bootstrap"

rerun_roi_report \
  "Bronchus" 10 4 \
  "${BRONCHUS_PILOT}" "${BRONCHUS_BENCH}" "${BRONCHUS_RANDOM}" "${BRONCHUS_V3}" \
  "${BRONCHUS_VIS}" "${BRONCHUS_REPORT}" \
  "Bronchus stochastic v3 symmetric changed-region loss; patient-cluster bootstrap"

echo
echo "Done. No probe inference was rerun; only pooled/per-probe statistics and reports were regenerated."
