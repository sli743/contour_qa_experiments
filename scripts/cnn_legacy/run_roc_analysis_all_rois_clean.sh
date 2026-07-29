#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run clean ROC analysis for every ROI present in the input CSV
# ============================================================
#
# Run from:
#   /home/shaohan/VariantVAE
#
# Outputs:
#   ${OUT_ROOT}/<ROI>_clean/
#       tables/
#       figures/
#       report/
#
# By default, ROI names are read directly from the CSV. To run a selected
# subset, set ROI_LIST as a space-separated list, for example:
#
#   ROI_LIST="Lung_L Lung_R Heart Liver" bash run_roc_analysis_all_rois_clean.sh
#
# Shift is excluded by default because rigid translation preserves volume and
# therefore is not meaningfully selected by a volume-change threshold.

SCRIPT="${SCRIPT:-roc_analysis_clean.py}"

FDVC_ROOT="${FDVC_ROOT:-../radformation-data/probe_auc_sensitivity_assd/first_detectable_volume_custom/}"
CSV="${CSV:-${FDVC_ROOT}/input_with_volume_and_scores.csv}"
OUT_ROOT="${OUT_ROOT:-${FDVC_ROOT}/roc_analysis_volume_gt20_all_rois}"

VOLUME_THRESHOLD_PCT="${VOLUME_THRESHOLD_PCT:-20}"
N_BOOT="${N_BOOT:-1000}"
SEED="${SEED:-123}"
INCLUDE_SHIFT="${INCLUDE_SHIFT:-0}"

# Leave empty to use the default metrics in roc_analysis_clean.py.
METRICS="${METRICS:-}"

# Leave empty to discover all ROI values from the CSV.
ROI_LIST="${ROI_LIST:-}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: missing analysis script: ${SCRIPT}"
  exit 1
fi

if [[ ! -f "${CSV}" ]]; then
  echo "ERROR: missing input CSV: ${CSV}"
  exit 1
fi

mkdir -p "${OUT_ROOT}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

if [[ -z "${ROI_LIST}" ]]; then
  ROI_LIST="$(
    python - "${CSV}" <<'PY'
import sys
import pandas as pd

csv_path = sys.argv[1]
df = pd.read_csv(csv_path, usecols=lambda c: c == "roi")
if "roi" not in df.columns:
    raise SystemExit("ERROR: input CSV does not contain an 'roi' column.")

rois = sorted(
    {
        str(x).strip()
        for x in df["roi"].dropna().tolist()
        if str(x).strip() and str(x).strip().lower() != "nan"
    },
    key=lambda x: x.lower(),
)

if not rois:
    raise SystemExit("ERROR: no ROI names found in the CSV.")

print("\n".join(rois))
PY
  )"
fi

mapfile -t ROIS < <(printf '%s\n' "${ROI_LIST}" | tr ' ' '\n' | sed '/^[[:space:]]*$/d')

if [[ "${#ROIS[@]}" -eq 0 ]]; then
  echo "ERROR: no ROIs selected."
  exit 1
fi

echo "Input CSV:        ${CSV}"
echo "Output root:      ${OUT_ROOT}"
echo "Volume threshold: > ${VOLUME_THRESHOLD_PCT}%"
echo "Bootstrap runs:   ${N_BOOT}"
echo "ROIs (${#ROIS[@]}):"
printf '  %s\n' "${ROIS[@]}"
echo

FAILED=()
SUCCEEDED=()

for ROI in "${ROIS[@]}"; do
  SAFE_ROI="$(printf '%s' "${ROI}" | sed -E 's/[^A-Za-z0-9_.+-]+/_/g')"
  OUT_DIR="${OUT_ROOT}/${SAFE_ROI}_clean"
  LOG_FILE="${LOG_DIR}/${SAFE_ROI}.log"

  echo "============================================================"
  echo "Running ROI: ${ROI}"
  echo "Output:      ${OUT_DIR}"
  echo "Log:         ${LOG_FILE}"
  echo "============================================================"

  CMD=(
    python "${SCRIPT}"
    --csvs "${CSV}"
    --out_dir "${OUT_DIR}"
    --roi "${ROI}"
    --volume_threshold_pct "${VOLUME_THRESHOLD_PCT}"
    --n_boot "${N_BOOT}"
    --seed "${SEED}"
  )

  if [[ -n "${METRICS}" ]]; then
    # shellcheck disable=SC2206
    METRIC_ARGS=(${METRICS})
    CMD+=(--metrics "${METRIC_ARGS[@]}")
  fi

  if [[ "${INCLUDE_SHIFT}" == "1" ]]; then
    CMD+=(--include_shift)
  fi

  if "${CMD[@]}" > >(tee "${LOG_FILE}") 2>&1; then
    SUCCEEDED+=("${ROI}")
  else
    FAILED+=("${ROI}")
    echo "WARNING: ROC analysis failed for ${ROI}."
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

# Aggregate the per-ROI output tables.
python aggregate_clean_roc_all_rois.py \
  --root "${OUT_ROOT}" \
  --out_dir "${OUT_ROOT}/combined"

echo
echo "============================================================"
echo "All-ROI clean ROC run complete"
echo "============================================================"
echo "Succeeded (${#SUCCEEDED[@]}):"
printf '  %s\n' "${SUCCEEDED[@]:-}"
echo
echo "Failed (${#FAILED[@]}):"
printf '  %s\n' "${FAILED[@]:-}"
echo
echo "Combined outputs:"
echo "  ${OUT_ROOT}/combined"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  printf '%s\n' "${FAILED[@]}" > "${OUT_ROOT}/failed_rois.txt"
  echo "Failed ROI list written to:"
  echo "  ${OUT_ROOT}/failed_rois.txt"
fi
