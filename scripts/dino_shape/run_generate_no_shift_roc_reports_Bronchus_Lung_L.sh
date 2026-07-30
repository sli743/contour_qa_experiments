#!/usr/bin/env bash
set -euo pipefail

# Rebuild three-model ROC reports from existing probe-evaluation CSVs.
# Shift probes are deliberately excluded: compare_dino_shape_roc_reports excludes
# shifts by default unless --include_shift is explicitly supplied.

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
CONDA_ENV="${CONDA_ENV:-3DINO}"
N_BOOT="${N_BOOT:-2000}"
SEED="${SEED:-88}"

LUNG_THRESHOLD_PCT="${LUNG_THRESHOLD_PCT:-20}"
BRONCHUS_THRESHOLD_PCT="${BRONCHUS_THRESHOLD_PCT:-10}"

LUNG_PILOT_CONFIG="${LUNG_PILOT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Lung_L_probe_eval.yaml}"
LUNG_BENCHMARK_CONFIG="${LUNG_BENCHMARK_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1_eval.yaml}"
LUNG_RANDOM_CONFIG="${LUNG_RANDOM_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Lung_L_v1_eval.yaml}"
LUNG_PILOT_CSV="${LUNG_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Lung_L/probe_eval_sweep/probe_eval_sweep_detailed.csv}"
LUNG_BENCHMARK_CSV="${LUNG_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
LUNG_RANDOM_CSV="${LUNG_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Lung_L_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
LUNG_OUT_DIR="${LUNG_OUT_DIR:-/scratch/shaohan/contour_QA/roc_comparison/dino_shape_three_models_Lung_L_no_shift}"

BRONCHUS_PILOT_CONFIG="${BRONCHUS_PILOT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_Bronchus_probe_eval.yaml}"
BRONCHUS_BENCHMARK_CONFIG="${BRONCHUS_BENCHMARK_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Bronchus_v1_eval.yaml}"
BRONCHUS_RANDOM_CONFIG="${BRONCHUS_RANDOM_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/dino_shape_random_morphology_Bronchus_v1_eval.yaml}"
BRONCHUS_PILOT_CSV="${BRONCHUS_PILOT_CSV:-/scratch/shaohan/contour_QA/dino_frozen/dino_shape_pilot_Bronchus/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
BRONCHUS_BENCHMARK_CSV="${BRONCHUS_BENCHMARK_CSV:-/scratch/shaohan/contour_QA/dino_denoising/dino_shape_denoising_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
BRONCHUS_RANDOM_CSV="${BRONCHUS_RANDOM_CSV:-/scratch/shaohan/contour_QA/dino_random_morphology/dino_shape_random_morphology_Bronchus_v1/probe_eval_sweep_test/probe_eval_sweep_detailed.csv}"
BRONCHUS_OUT_DIR="${BRONCHUS_OUT_DIR:-/scratch/shaohan/contour_QA/roc_comparison/dino_shape_three_models_Bronchus_no_shift}"

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { echo "ERROR: missing required file: $path" >&2; exit 2; }
}

run_roi() {
  local roi="$1"
  local threshold="$2"
  local pilot_csv="$3"
  local benchmark_csv="$4"
  local random_csv="$5"
  local pilot_config="$6"
  local benchmark_config="$7"
  local random_config="$8"
  local out_dir="$9"

  local pair_pilot_random="${out_dir}/pilot_vs_random_morphology"
  local pair_benchmark_random="${out_dir}/benchmark_upper_limit_vs_random_morphology"

  for required in \
    "$pilot_csv" "$benchmark_csv" "$random_csv" \
    "$pilot_config" "$benchmark_config" "$random_config"; do
    require_file "$required"
  done

  echo
  echo "============================================================"
  echo "ROI: $roi"
  echo "Positive class: absolute volume change > ${threshold}%"
  echo "Shift probes: excluded"
  echo "Output: $out_dir"
  echo "============================================================"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  # Do not add --include_shift. Its absence is the implemented exclusion.
  python -m analysis.roc.compare_dino_shape_roc_reports \
    --pilot_csv "$pilot_csv" \
    --denoising_csv "$random_csv" \
    --pilot_config "$pilot_config" \
    --denoising_config "$random_config" \
    --out_dir "$pair_pilot_random" \
    --roi "$roi" \
    --split test \
    --volume_threshold_pct "$threshold" \
    --clean_policy one_per_patient \
    --n_boot "$N_BOOT" \
    --seed "$SEED"

  python -m analysis.roc.compare_dino_shape_roc_reports \
    --pilot_csv "$benchmark_csv" \
    --denoising_csv "$random_csv" \
    --pilot_config "$benchmark_config" \
    --denoising_config "$random_config" \
    --out_dir "$pair_benchmark_random" \
    --roi "$roi" \
    --split test \
    --volume_threshold_pct "$threshold" \
    --clean_policy one_per_patient \
    --n_boot "$N_BOOT" \
    --seed "$SEED"

  python -m analysis.roc.summarize_three_dino_shape_models \
    --pilot_random_dir "$pair_pilot_random" \
    --benchmark_random_dir "$pair_benchmark_random" \
    --out_dir "$out_dir"

  # Fail loudly if a future code change accidentally includes shifts.
  python - "$pair_pilot_random" "$pair_benchmark_random" <<'PY'
import json
import sys
from pathlib import Path

for directory in map(Path, sys.argv[1:]):
    metadata_path = directory / "analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("include_shift") is not False:
        raise SystemExit(f"ERROR: shift exclusion audit failed: {metadata_path}")
    print(f"Verified shift exclusion: {metadata_path}")
PY

  if command -v pandoc >/dev/null 2>&1; then
    find "$out_dir" -type f -name '*.md' -print0 | while IFS= read -r -d '' md; do
      pandoc "$md" \
        --standalone \
        --toc \
        --embed-resources \
        --highlight-style=tango \
        --metadata title="$(basename "${md%.md}")" \
        -o "${md%.md}.html"
    done
  else
    echo "Pandoc not found; Markdown reports were generated."
  fi

  echo "Main report: ${out_dir}/report/three_model_ROC_comparison.md"
  [[ ! -f "${out_dir}/report/three_model_ROC_comparison.html" ]] || \
    echo "HTML report: ${out_dir}/report/three_model_ROC_comparison.html"
}

cd "$PROJECT_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

run_roi \
  "Lung_L" "$LUNG_THRESHOLD_PCT" \
  "$LUNG_PILOT_CSV" "$LUNG_BENCHMARK_CSV" "$LUNG_RANDOM_CSV" \
  "$LUNG_PILOT_CONFIG" "$LUNG_BENCHMARK_CONFIG" "$LUNG_RANDOM_CONFIG" \
  "$LUNG_OUT_DIR"

run_roi \
  "Bronchus" "$BRONCHUS_THRESHOLD_PCT" \
  "$BRONCHUS_PILOT_CSV" "$BRONCHUS_BENCHMARK_CSV" "$BRONCHUS_RANDOM_CSV" \
  "$BRONCHUS_PILOT_CONFIG" "$BRONCHUS_BENCHMARK_CONFIG" "$BRONCHUS_RANDOM_CONFIG" \
  "$BRONCHUS_OUT_DIR"

echo
echo "Completed no-shift three-model ROC reports for Lung_L and Bronchus."
