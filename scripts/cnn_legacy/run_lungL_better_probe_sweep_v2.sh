#!/usr/bin/env bash
set -euo pipefail

# Run severity-sweep "better evaluation" for current Lung_L AE.
# Run from /home/shaohan/VariantVAE.
# Copy eval_post_dae_probe_sitk_morph_sweep_v2.py into this directory first.

CONFIG_PATH="configs/label_ae_lungL_roiwise_1p5mm.yaml"
EVAL_SCRIPT="eval_post_dae_probe_sitk_morph_sweep_v2.py"

CUT_LEVELS="0,0.1,0.2,0.3,0.4,0.5"
LOCAL_FRAC_LEVELS="0,0.1,0.2,0.3,0.4,0.5"
VOXEL_LEVELS="0,1,2,3,4,5"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: Config not found: ${CONFIG_PATH}"
  exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: ${EVAL_SCRIPT} not found in current directory."
  exit 1
fi

python "${EVAL_SCRIPT}" \
  -c "${CONFIG_PATH}" \
  --cut_levels "${CUT_LEVELS}" \
  --local_frac_levels "${LOCAL_FRAC_LEVELS}" \
  --voxel_levels "${VOXEL_LEVELS}" \
  --drop_inf_in_plots

echo
echo "Done."
echo "Outputs:"
echo "  /scratch/shaohan/experiments/VariantVAE/label_ae_lungL_roiwise_1p5mm_alpha0.8_kld0.001_wmuniform/probe_eval_sweep/"
echo "Plots include:"
echo "  plots_by_probe_level/*dice_input_vs_target*"
echo "  plots_by_probe_level/*hd95_input_vs_target*"
echo "  plots_by_probe_level/severity_trend_dice_input_vs_target_mean.png"
echo "  plots_by_probe_level/severity_trend_hd95_input_vs_target_mean.png"
