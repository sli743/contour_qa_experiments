#!/usr/bin/env bash
set -euo pipefail

# Lung_R corruption x CT study, stage 1 of 2: calibrate Lung_R corruption, then
# prepare the contour-only study that every Lung_R model is built from.
#
#   1. Severity fit with exactly the Lung_L v3 method on Lung_R error fields
#      (configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v3method.yaml; CPU job).
#   2. Month-temporal prepare of V3_STUDY with the fit's best candidate config
#      (as Lung_L used best_candidate_training_config.yaml from its v3 fit): writes
#      experiment_metadata.json, seed-91 configs and the Month3 crops with CT.
#      Nothing is trained here.
#
# Stage 2, once the prepare job has finished, trains all five models for seeds
# 91, 92, 93 and runs the per-seed ablation:
#   ROI=Lung_R bash scripts/dino_shape/submit_lung_l_ct_contour_ablation_seeds.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
FIT_CONFIG="${FIT_CONFIG:-configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v3method.yaml}"
V3_STUDY="${V3_STUDY:-/scratch/shaohan/contour_QA/lung_r_nsrqa_month_temporal_family_validated_v1}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"

if [[ -s "${V3_STUDY}/experiment_metadata.json" && "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "ERROR: ${V3_STUDY} is already prepared. Pick another V3_STUDY or set ALLOW_EXISTING=1." >&2
  exit 2
fi

read -r EF INIT FIT_OUT < <(python - "${FIT_CONFIG}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["error_field_root"], c["initial_training_config"], c["output"]["directory"])
PY
)
for f in "${EF}/case_error_field_summary.csv" "${EF}/error_components.csv" "${INIT}"; do
  [[ -s "${f}" ]] || { echo "ERROR: missing ${f} (run submit_lung_r_nsrqa_severity_calibration.sh first)" >&2; exit 2; }
done
mkdir -p "${FIT_OUT}" "${V3_STUDY}/logs"
jobid() { printf '%s' "$1" | cut -d';' -f1; }

FIT="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name=lungR_sevfit_v3method \
  --export="ALL,REPO=${PROJECT_ROOT},CONFIG=${FIT_CONFIG}" \
  --output="${FIT_OUT}/slurm-%x-%j.out" --error="${FIT_OUT}/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_synthetic_severity_fit_family_validated.job)")"

# The fit exits 0 even when the family-validity audit fails; it always writes
# best_candidate_training_config.yaml, which is what Lung_L v3 used too.
PREP="$(jobid "$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name=lungR_ctc_prep \
  --dependency="afterok:${FIT}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},ROI=Lung_R,STUDY_ROOT=${V3_STUDY},SEED=91,CALIBRATED_CONFIG=${FIT_OUT}/best_candidate_training_config.yaml" \
  --output="${V3_STUDY}/logs/slurm-%x-%j.out" --error="${V3_STUDY}/logs/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_month_temporal_ablation_prepare.job)")"

cat <<EOF
Submitted Lung_R stage 1
  severity fit (v3 method): ${FIT}   -> ${FIT_OUT}
  contour-only prepare:     ${PREP}  -> ${V3_STUDY}

Check the fit: ${FIT_OUT}/calibration_status.json, real_vs_synthetic_quantiles.csv, report.md
When job ${PREP} has finished, start stage 2 (all five models, seeds 91 92 93):
  ROI=Lung_R bash scripts/dino_shape/submit_lung_l_ct_contour_ablation_seeds.sh
EOF
