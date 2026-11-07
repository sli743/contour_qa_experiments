#!/usr/bin/env bash
set -euo pipefail

# Lung_R corruption-severity estimation with the same method that produced the
# Lung_L severities used by lung_l_nsrqa_corruption_ablation_month_temporal_v1
# (lung_l_error_field_characterization_v1 -> lung_l_nsrqa_severity_calibration_v1):
#
#   1. analysis.characterize_lung_error_fields   (approved vs autoseg error fields,
#      Month1+Month2 temporal training1 only)
#   2. analysis.calibrate_lung_l_nsrqa_severity  (P10/P90 of those fields ->
#      severity ranges; families/probabilities copied from the Lung_L v3 config)
#
# Stage 2 waits for stage 1 (afterok). Training is NOT launched here; after
# checking severity_calibration.csv / report.md, train with
#   CALIBRATED_CONFIG=${CAL_OUT}/recommended_training_config.yaml \
#   STUDY_ROOT=/scratch/shaohan/contour_QA/lung_r_nsrqa_corruption_ablation_month_temporal_rcal_v1 \
#   bash scripts/dino_shape/submit_lung_r_nsrqa_month_temporal_ablation.sh

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
EF_CONFIG="${EF_CONFIG:-configs/dino_shape/lung_r_error_field_characterization.yaml}"
CAL_CONFIG="${CAL_CONFIG:-configs/dino_shape/lung_r_nsrqa_severity_calibration.yaml}"
cd "${PROJECT_ROOT}"

EF_OUT="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${EF_CONFIG}")"
CAL_OUT="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output"]["directory"])' "${CAL_CONFIG}")"
mkdir -p "${EF_OUT}" "${CAL_OUT}"

jobid() { printf '%s' "$1" | cut -d';' -f1; }

if [[ "${SKIP_ERROR_FIELD:-0}" == "1" ]]; then
  [[ -s "${EF_OUT}/case_error_field_summary.csv" ]] || { echo "No ${EF_OUT}/case_error_field_summary.csv" >&2; exit 2; }
  DEP=()
  echo "Reusing existing error fields: ${EF_OUT}"
else
  EF_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
    --job-name=lungR_errfield \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONFIG=${PROJECT_ROOT}/${EF_CONFIG}" \
    --output="${EF_OUT}/slurm-%x-%j.out" --error="${EF_OUT}/slurm-%x-%j.err" \
    jobs/lung_l_error_field_characterization.job)"
  EF="$(jobid "${EF_RAW}")"
  DEP=(--dependency="afterok:${EF}")
  echo "Submitted Lung_R error-field characterization: ${EF}"
fi

CAL_RAW="$(sbatch --parsable -p hugo --account=geoffrey_hugo \
  --job-name=lungR_sevcal "${DEP[@]}" \
  --export="ALL,REPO=${PROJECT_ROOT},CONFIG=${CAL_CONFIG}" \
  --output="${CAL_OUT}/slurm-%x-%j.out" --error="${CAL_OUT}/slurm-%x-%j.err" \
  jobs/lung_l_nsrqa_severity_calibration.job)"
echo "Submitted Lung_R severity calibration: $(jobid "${CAL_RAW}")"
echo
echo "Outputs:"
echo "  error fields: ${EF_OUT}"
echo "  severities:   ${CAL_OUT}/severity_calibration.csv"
echo "  train config: ${CAL_OUT}/recommended_training_config.yaml"
