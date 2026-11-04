#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_study}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
REG_ROOT="${REG_ROOT:-${STUDY_ROOT}/actual_autoseg_regression}"
N_BOOT="${N_BOOT:-5000}"
SEED="${SEED:-88}"
AUTOSEG_CENTER_METHOD="${AUTOSEG_CENTER_METHOD:-foreground_centroid}"
ROI_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
[[ -n "${ROI_INDEX}" ]] || { echo "Usage: $0 ROI_INDEX" >&2; exit 2; }

MANIFEST="${STUDY_ROOT}/manifests/active_rois.tsv"
line="$(awk -F '\t' -v idx="${ROI_INDEX}" 'NR>1 && $1==idx {print; exit}' "${MANIFEST}")"
[[ -n "${line}" ]] || { echo "No active ROI index ${ROI_INDEX}" >&2; exit 2; }
line="${line%$'\r'}"
IFS=$'\t' read -r roi_index registry_index ROI data_group profile GLOBAL_THRESHOLD_PCT LOCAL_DEPTH_MM PILOT_CONFIG BENCH_CONFIG RANDOM_CONFIG NSRQA_CONFIG <<< "${line}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
unset PYTORCH_CUDA_ALLOC_CONF || true

NSRQA_CKPT="${STUDY_ROOT}/models/nsrqa/dino_shape_NSRQA_${ROI}_allroi/checkpoints/best_decoder.pth"
[[ -s "${NSRQA_CKPT}" ]] || { echo "Missing NSR-QA checkpoint: ${NSRQA_CKPT}" >&2; exit 3; }

echo "[${ROI}] Step 1/4: preprocess actual autosegmentations in both localization modes"
python -m evaluation.actual_autoseg.preprocess_actual_autoseg \
  --config "${NSRQA_CONFIG}" \
  --roi "${ROI}" \
  --group "${data_group}" \
  --raw-data-root "${RAW_DATA_ROOT}" \
  --output-root "${REG_ROOT}" \
  --mode both \
  --center-method "${AUTOSEG_CENTER_METHOD}" \
  --splits training1 validation1 testing1

PRIMARY_CSV="${REG_ROOT}/metrics/${ROI}/autoseg_centered_metrics.csv"
ORACLE_CSV="${REG_ROOT}/metrics/${ROI}/approved_reference_metrics.csv"
mkdir -p "$(dirname "${PRIMARY_CSV}")"

echo "[${ROI}] Step 2/4: NSR-QA inference, candidate/autoseg-centered localization (primary)"
python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
  --config "${NSRQA_CONFIG}" \
  --checkpoint "${NSRQA_CKPT}" \
  --input-root "${REG_ROOT}" \
  --roi "${ROI}" \
  --mode autoseg_centered \
  --output-csv "${PRIMARY_CSV}"

echo "[${ROI}] Step 3/4: NSR-QA inference, approved-reference localization (oracle sensitivity)"
python -m evaluation.actual_autoseg.eval_actual_autoseg_nsrqa \
  --config "${NSRQA_CONFIG}" \
  --checkpoint "${NSRQA_CKPT}" \
  --input-root "${REG_ROOT}" \
  --roi "${ROI}" \
  --mode approved_reference \
  --output-csv "${ORACLE_CSV}"

echo "[${ROI}] Step 4/4: regression analysis"
python -m analysis.analyze_actual_autoseg_regression \
  --roi "${ROI}" \
  --metrics "autoseg_centered=${PRIMARY_CSV}" \
  --metrics "approved_reference=${ORACLE_CSV}" \
  --output-dir "${REG_ROOT}/regression/${ROI}" \
  --n-boot "${N_BOOT}" \
  --seed "${SEED}"

echo "Done ${ROI}"
echo "  Visual cases: ${REG_ROOT}/{autoseg_centered,approved_reference}/${ROI}/<split>/patient_<id>/"
echo "  Metrics:      ${REG_ROOT}/metrics/${ROI}/"
echo "  Regression:   ${REG_ROOT}/regression/${ROI}/"
