#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate the family-validated Lung_L NSR-QA corruption model on the
# leakage-safe Month1+Month2 -> Month3 temporal split. This wrapper intentionally
# submits only task 0 (the NSR-QA corruption arm); it does not retrain the
# no-corruption baseline.

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/lung_l_nsrqa_month_temporal_family_validated_v1}"
TEMPORAL_STUDY_ROOT="${TEMPORAL_STUDY_ROOT:-/scratch/shaohan/contour_QA/all_roi_NSRQA_month_temporal_study}"
SPLIT_CONFIG="${SPLIT_CONFIG:-${PROJECT_ROOT}/configs/dino_shape/month_temporal_nsrqa_split.yaml}"
CALIBRATED_CONFIG="${CALIBRATED_CONFIG:-}"
INSPECTION_SOURCE="${INSPECTION_SOURCE:-/scratch/shaohan/contour_QA/lung_l_nsrqa_synthetic_inspection_bank_v1/inspection_source.json}"
FAMILY_CAL_DIR="${FAMILY_CAL_DIR:-/scratch/shaohan/contour_QA/lung_l_nsrqa_synthetic_severity_fit_v3_family_validated}"
SEED="${SEED:-91}"

cd "${PROJECT_ROOT}"

required=(
  analysis/prepare_lung_l_nsrqa_month_temporal_ablation.py
  jobs/lung_l_nsrqa_month_temporal_ablation_prepare.job
  jobs/lung_l_nsrqa_month_temporal_ablation_train.job
  jobs/lung_l_nsrqa_month_temporal_ablation_eval.job
  scripts/dino_shape/prepare_lung_l_nsrqa_month_temporal_ablation.sh
)
for path in "${required[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing Month-temporal ablation component: ${PROJECT_ROOT}/${path}" >&2
    echo "Apply the Month1+2 -> Month3 ablation patch first." >&2
    exit 2
  fi
done

# Prefer the exact config that generated the synthetic contours that were
# visually inspected. This avoids guessing whether the accepted calibration
# artifact was named best_fit_training_config.yaml or
# best_candidate_training_config.yaml.
CONFIG_SOURCE="explicit CALIBRATED_CONFIG"
INSPECTION_VALIDATED="unknown"
if [[ -z "${CALIBRATED_CONFIG}" ]]; then
  if [[ -s "${INSPECTION_SOURCE}" ]]; then
    INSPECTION_CONFIG="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_config"])' "${INSPECTION_SOURCE}")"
    INSPECTION_VALIDATED="$(python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1])).get("validated_training_config", False))).lower())' "${INSPECTION_SOURCE}")"
    CALIBRATED_CONFIG="${INSPECTION_CONFIG}"
    CONFIG_SOURCE="synthetic inspection bank (${INSPECTION_SOURCE})"
  elif [[ -s "${FAMILY_CAL_DIR}/best_fit_training_config.yaml" ]]; then
    CALIBRATED_CONFIG="${FAMILY_CAL_DIR}/best_fit_training_config.yaml"
    CONFIG_SOURCE="family-validity validated fit"
    INSPECTION_VALIDATED="true"
  elif [[ -s "${FAMILY_CAL_DIR}/best_candidate_training_config.yaml" ]]; then
    CALIBRATED_CONFIG="${FAMILY_CAL_DIR}/best_candidate_training_config.yaml"
    CONFIG_SOURCE="family-validity best candidate (not independently validated)"
    INSPECTION_VALIDATED="false"
  fi
fi

if [[ -z "${CALIBRATED_CONFIG}" || ! -s "${CALIBRATED_CONFIG}" ]]; then
  echo "ERROR: artificial-corruption training config could not be resolved." >&2
  echo "Checked:" >&2
  echo "  inspection record: ${INSPECTION_SOURCE}" >&2
  echo "  validated fit:      ${FAMILY_CAL_DIR}/best_fit_training_config.yaml" >&2
  echo "  best candidate:     ${FAMILY_CAL_DIR}/best_candidate_training_config.yaml" >&2
  echo >&2
  echo "If you want a different config, pass CALIBRATED_CONFIG=/path/to/config.yaml." >&2
  exit 2
fi

echo "Resolved artificial-corruption config:"
echo "  ${CALIBRATED_CONFIG}"
echo "Config source: ${CONFIG_SOURCE}"
if [[ "${INSPECTION_VALIDATED}" == "false" ]]; then
  echo "NOTE: the accepted inspection bank was generated from a best-candidate config"
  echo "      that did not carry the independent family-validity PASS flag."
  echo "      Proceeding because this wrapper intentionally reproduces the inspected contours."
fi

mkdir -p "${STUDY_ROOT}/logs"

# Record the exact source config and show the relevant stochastic-corruption
# mixture before handing the job to Slurm. This is intentionally not a generic
# rotation/zoom augmentation pipeline; the DINO-shape trainer consumes binary
# masks and applies the NSR-QA corruption generator directly.
python - "${CALIBRATED_CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
spec = cfg.get("stochastic_region_dropout", {})
print("Family-validated training source:", p)
print("  clean_probability:", spec.get("clean_probability"))
print("  branch_probabilities:", spec.get("branch_probabilities"))
rf = cfg.get("reference_frame_training") or {}
jitter = rf.get("fov_jitter") or {}
print("  candidate_centered reference frame:", bool(rf.get("candidate_centered", False)))
print("  FOV jitter enabled:", bool(jitter.get("enabled", False)))
print("  conventional rotation/zoom/flip augmentation: not implemented in this trainer")
PY

EXPORTS="ALL,PROJECT_ROOT=${PROJECT_ROOT},STUDY_ROOT=${STUDY_ROOT},TEMPORAL_STUDY_ROOT=${TEMPORAL_STUDY_ROOT},SPLIT_CONFIG=${SPLIT_CONFIG},CALIBRATED_CONFIG=${CALIBRATED_CONFIG},SEED=${SEED}"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

PREP_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%j.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_month_temporal_ablation_prepare.job)"
PREP="$(jobid "${PREP_RAW}")"
echo "Submitted temporal preparation: ${PREP}"

# prepare_lung_l_nsrqa_month_temporal_ablation.py always writes task 0 as the
# NSR-QA corruption arm and task 1 as the identity baseline. Only task 0 is
# submitted here because this wrapper is for the new corruption model itself.
TRAIN_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${PREP}" \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0" \
  jobs/lung_l_nsrqa_month_temporal_ablation_train.job)"
TRAIN="$(jobid "${TRAIN_RAW}")"
echo "Submitted family-validated training: ${TRAIN}"

EVAL_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --dependency="afterok:${TRAIN}" \
  --export="${EXPORTS}" \
  --output="${STUDY_ROOT}/logs/slurm-%x-%A_%a.out" \
  --error="${STUDY_ROOT}/logs/slurm-%x-%A_%a.err" \
  --array="0" \
  jobs/lung_l_nsrqa_month_temporal_ablation_eval.job)"
EVAL="$(jobid "${EVAL_RAW}")"
echo "Submitted Month3 real-autoseg evaluation: ${EVAL}"

cat <<EOF2

Family-validated Lung_L temporal run submitted
  prepare:  ${PREP}
  train:    ${TRAIN}
  evaluate: ${EVAL}

Study root:
  ${STUDY_ROOT}

Artificial-corruption config:
  ${CALIBRATED_CONFIG}

Split:
  Month1+Month2 = development training/validation
  Month3        = locked real-autoseg test

Seed:
  ${SEED}

Only the NSR-QA corruption arm is trained by this wrapper.
EOF2
