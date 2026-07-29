#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run severity-sweep "better evaluation" for current Lung_L AE
# ============================================================
# Run from:
#   /home/shaohan/VariantVAE
#
# Required existing script:
#   eval_post_dae_probe_sitk_morph_sweep.py
#
# This assumes the model has already been trained using:
#   configs/label_ae_lungL_roiwise_1p5mm.yaml

CONFIG_PATH="configs/label_ae_lungL_roiwise_1p5mm.yaml"

# Severity grids requested:
#   cut posterior/upper/lower ratios: 0% to 50%
#   local blob/bite frac: 0 to 0.5, no min block
#   shift/erode/dilate voxels: 0 to 5
CUT_LEVELS="0,0.1,0.2,0.3,0.4,0.5"
LOCAL_FRAC_LEVELS="0,0.1,0.2,0.3,0.4,0.5"
VOXEL_LEVELS="0,1,2,3,4,5"

SAVE_VIS=0
DROP_INF_IN_PLOTS=1

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Config not found: ${CONFIG_PATH}"
  echo "Run the Lung_L training wrapper first, or edit CONFIG_PATH in this script."
  exit 1
fi

if [[ ! -f "eval_post_dae_probe_sitk_morph_sweep.py" ]]; then
  echo "ERROR: eval_post_dae_probe_sitk_morph_sweep.py not found in current directory."
  echo "Copy it into /home/shaohan/VariantVAE first."
  exit 1
fi

CMD=(
  python eval_post_dae_probe_sitk_morph_sweep.py
  -c "${CONFIG_PATH}"
  --cut_levels "${CUT_LEVELS}"
  --local_frac_levels "${LOCAL_FRAC_LEVELS}"
  --voxel_levels "${VOXEL_LEVELS}"
)

if [[ "${SAVE_VIS}" == "1" ]]; then
  CMD+=(--save_vis)
fi

if [[ "${DROP_INF_IN_PLOTS}" == "1" ]]; then
  CMD+=(--drop_inf_in_plots)
fi

echo "============================================================"
echo "Running Lung_L severity-sweep probe evaluation"
echo "Config:            ${CONFIG_PATH}"
echo "Cut levels:        ${CUT_LEVELS}"
echo "Local frac levels: ${LOCAL_FRAC_LEVELS}"
echo "Voxel levels:      ${VOXEL_LEVELS}"
echo "No min block:      blob_min_block=0, bite_min_block=0"
echo "============================================================"
echo "${CMD[@]}"
echo

"${CMD[@]}"

echo
echo "Done."
echo "Expected outputs are inside the experiment folder:"
echo "  /scratch/shaohan/experiments/VariantVAE/label_ae_lungL_roiwise_1p5mm_alpha0.8_kld0.001_wmuniform/probe_eval_sweep/"
echo
echo "Key files:"
echo "  probe_eval_sweep_detailed.csv"
echo "  probe_eval_sweep_summary.csv"
echo "  plots_by_probe_level/"
