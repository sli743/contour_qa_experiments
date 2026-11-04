#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
STUDY_ROOT="${STUDY_ROOT:-/scratch/shaohan/contour_QA/spinalcord_large_fov_NSRQA_study}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/home/shaohan/radformation-data/preprocessed_roiwise_allrois_1p5mm_nozoom}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/home/shaohan/radformation-data/data}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/dino_shape/spinalcord_large_fov_nsrqa_registry.yaml}"
SPLIT_CONFIG="${SPLIT_CONFIG:-}"
# These are minima. With AUTO_SIZE=1 the preparer expands any axis as needed
# from approved SpinalCord bboxes in SIZING_SPLITS plus MARGIN_*_MM per side.
CROP_X="${CROP_X:-96}"
CROP_Y="${CROP_Y:-96}"
CROP_Z="${CROP_Z:-384}"
SPACING_MM="${SPACING_MM:-1.5}"
AUTO_SIZE="${AUTO_SIZE:-1}"
SIZING_SPLITS="${SIZING_SPLITS:-training1 validation1}"
MARGIN_X_MM="${MARGIN_X_MM:-15}"
MARGIN_Y_MM="${MARGIN_Y_MM:-15}"
MARGIN_Z_MM="${MARGIN_Z_MM:-15}"
ROUND_TO="${ROUND_TO:-16}"
FRESH_DATA="${FRESH_DATA:-0}"
GENERATE_EXAMPLES="${GENERATE_EXAMPLES:-1}"

cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/dino_shape/activate_3dino_env.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

extra=()
[[ -n "${SPLIT_CONFIG}" ]] && extra+=(--split-config "${SPLIT_CONFIG}")
[[ "${FRESH_DATA}" == "1" ]] && extra+=(--overwrite-data)
if [[ "${AUTO_SIZE}" == "1" ]]; then
  read -r -a sizing_split_args <<< "${SIZING_SPLITS}"
  extra+=(
    --auto-size-from-approved
    --sizing-splits "${sizing_split_args[@]}"
    --margin-mm-xyz "${MARGIN_X_MM}" "${MARGIN_Y_MM}" "${MARGIN_Z_MM}"
    --round-to "${ROUND_TO}"
  )
fi

python scripts/dino_shape/prepare_spinalcord_large_fov_nsrqa_study.py \
  --study-root "${STUDY_ROOT}" \
  --source-data-root "${SOURCE_DATA_ROOT}" \
  --raw-data-root "${RAW_DATA_ROOT}" \
  --registry "${REGISTRY}" \
  --crop-size-xyz "${CROP_X}" "${CROP_Y}" "${CROP_Z}" \
  --spacing-mm "${SPACING_MM}" \
  "${extra[@]}"

if [[ "${GENERATE_EXAMPLES}" == "1" ]]; then
  REGISTRY="${REGISTRY}" STUDY_ROOT="${STUDY_ROOT}" \
    bash scripts/dino_shape/generate_all_roi_training_examples.sh
fi

echo
echo "Before training inspect:"
echo "  ${STUDY_ROOT}/manifests/spinalcord_fov_sizing_audit.csv"
echo "  ${STUDY_ROOT}/manifests/spinalcord_large_fov_coverage_audit.csv"
echo "  ${STUDY_ROOT}/manifests/spinalcord_large_fov_spec.json"
echo "  ${STUDY_ROOT}/training_corruption_examples/SpinalCord/"
