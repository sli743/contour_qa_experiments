#!/usr/bin/env bash
set -euo pipefail

# Lung_R realized-metric severity fit (v4): the Lung_L v3 family-validated fit
# method (lung_l_nsrqa_synthetic_severity_fit_v3_family_validated), applied to
# Lung_R with deeper dropout targets and HD99 in the objective. See
# configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v4.yaml.
#
# Needs (from submit_lung_r_nsrqa_severity_calibration.sh):
#   /scratch/shaohan/contour_QA/lung_r_error_field_characterization_v1
#   /scratch/shaohan/contour_QA/lung_r_nsrqa_severity_calibration_v1/recommended_training_config.yaml
#
# Training is NOT launched. Afterwards (see LUNG_R_NSRQA_SEVERITY_CALIBRATION.md):
#   CALIBRATED_CONFIG=<fit dir>/best_fit_training_config.yaml  (or best_candidate_... if the
#   family-validity audit failed) STUDY_ROOT=... bash scripts/dino_shape/submit_lung_r_nsrqa_month_temporal_ablation.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
CONFIG="${CONFIG:-configs/dino_shape/lung_r_nsrqa_synthetic_severity_fit_v4.yaml}"
cd "${PROJECT_ROOT}"

read -r EF INIT OUT < <(python - "${CONFIG}" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["error_field_root"], c["initial_training_config"], c["output"]["directory"])
PY
)
for f in "${EF}/case_error_field_summary.csv" "${EF}/error_components.csv" "${INIT}"; do
  [[ -s "${f}" ]] || { echo "ERROR: missing ${f}" >&2; echo "Run submit_lung_r_nsrqa_severity_calibration.sh first." >&2; exit 2; }
done
mkdir -p "${OUT}"
TAG="$(basename "${CONFIG}" .yaml)"; TAG="${TAG##*_fit_}"   # v4, v4b, ...

JOB="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name="lungR_sevfit_${TAG}" \
  --export="ALL,REPO=${PROJECT_ROOT},CONFIG=${CONFIG}" \
  --output="${OUT}/slurm-%x-%j.out" --error="${OUT}/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_synthetic_severity_fit_family_validated.job)"
echo "Submitted Lung_R severity fit ${TAG}: ${JOB%%;*}"
echo "Output: ${OUT}"
echo "  calibration_status.json (calibration_passed), real_vs_synthetic_quantiles.csv, report.md"
