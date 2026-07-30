#!/usr/bin/env bash
set -euo pipefail

DINO_SRC="${DINO_SRC:-/home/shaohan/3DINO}"
VAE_SRC="${VAE_SRC:-/home/shaohan/VariantVAE}"
DEST="${DEST:-/home/shaohan/contour_qa_experiments}"
DRY_RUN="${DRY_RUN:-0}"

for src in "$DINO_SRC" "$VAE_SRC"; do
  [[ -d "$src" ]] || { echo "ERROR: source directory not found: $src" >&2; exit 1; }
done

RSYNC_FLAGS=(-a --update --itemize-changes)
[[ "$DRY_RUN" == "1" ]] && RSYNC_FLAGS+=(--dry-run)

mkdir_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then echo "mkdir -p $*"; else mkdir -p "$@"; fi
}
copy_file() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    mkdir_cmd "$dst"
    rsync "${RSYNC_FLAGS[@]}" "$src" "$dst/"
  else
    echo "WARNING: missing file, skipped: $src" >&2
  fi
}
copy_dir() {
  local src="$1" dst="$2"
  if [[ -d "$src" ]]; then
    mkdir_cmd "$dst"
    rsync "${RSYNC_FLAGS[@]}" \
      --exclude='__pycache__/' --exclude='*.pyc' --exclude='wandb/' --exclude='.git/' \
      "$src/" "$dst/"
  else
    echo "WARNING: missing directory, skipped: $src" >&2
  fi
}

for d in \
  "$DEST/models/dino_shape" "$DEST/models/cnn_legacy" "$DEST/training" \
  "$DEST/evaluation/probes" "$DEST/evaluation/metrics" \
  "$DEST/analysis/roc" "$DEST/analysis/fdvc" "$DEST/analysis/plots" \
  "$DEST/configs/dino_shape" "$DEST/configs/cnn/roiwise_1p5mm_nozoom_allrois" \
  "$DEST/scripts/dino_shape" "$DEST/scripts/cnn_legacy" \
  "$DEST/legacy/variantvae_network" "$DEST/legacy/variantvae_dataset" \
  "$DEST/slicer" "$DEST/manifests"
do
  mkdir_cmd "$d"
done

# Frozen-3DINO shape pilot only; CT-reconstruction files are deliberately excluded.
for f in \
  dino_shape_autoencoder.py \
  train_dino_shape_decoder.py \
  eval_dino_shape_clean.py \
  eval_dino_shape_probes_visual.py \
  dino_shape_pilot_Lung_L.yaml \
  run_dino_shape_pilot_Lung_L.sh \
  test_3dino_checkpoint_features.py
do
  case "$f" in
    dino_shape_autoencoder.py|test_3dino_checkpoint_features.py) dst="$DEST/models/dino_shape" ;;
    train_dino_shape_decoder.py) dst="$DEST/training" ;;
    eval_dino_shape_clean.py) dst="$DEST/evaluation" ;;
    eval_dino_shape_probes_visual.py) dst="$DEST/evaluation/probes" ;;
    dino_shape_pilot_Lung_L.yaml) dst="$DEST/configs/dino_shape" ;;
    run_dino_shape_pilot_Lung_L.sh) dst="$DEST/scripts/dino_shape" ;;
  esac
  copy_file "$DINO_SRC/$f" "$dst"
done

# CNN model and dataset context for later import repair.
copy_dir "$VAE_SRC/network" "$DEST/legacy/variantvae_network"
copy_dir "$VAE_SRC/dataset" "$DEST/legacy/variantvae_dataset"
for f in main.py training.py training_dicebce_allmodels.py models_vqvae_fixed.py; do
  copy_file "$VAE_SRC/$f" "$DEST/models/cnn_legacy"
done

# Previous probe-generation/evaluation code.
for f in \
  eval_post_dae_probe.py \
  eval_post_dae_probe_sitk_morph.py \
  eval_post_dae_probe_sitk_morph_sweep.py \
  eval_post_dae_probe_sitk_morph_sweep_v2.py \
  eval_post_dae_probe_sitk_morph_sweep_assd.py \
  eval_post_dae_probe_sitk_morph_sweep_assd_vis.py
do
  copy_file "$VAE_SRC/$f" "$DEST/evaluation/probes"
done

# Detector metrics and model selection.
for f in \
  compute_probe_auc_sensitivity.py \
  compute_probe_auc_sensitivity_assd.py \
  select_best_detector_per_roi.py \
  select_best_detector_per_roi_assd.py
do
  copy_file "$VAE_SRC/$f" "$DEST/evaluation/metrics"
done

# ROC analysis.
for f in \
  roc_analysis_clean.py \
  aggregate_clean_roc_all_rois.py \
  roc_analysis_lungL_volume_gt20.py \
  roc_analysis_lungL_volume_gt20_ops.py \
  roc_analysis_lungL_volume_gt20_ops_fixed.py \
  patch_lungL_roc_report_with_figures.py \
  pack_roc_report_assets_html.py
do
  copy_file "$VAE_SRC/$f" "$DEST/analysis/roc"
done

# FDVC analysis.
for f in compute_first_detectable_volume_change.py analyze_roi_probe_fdvc.py; do
  copy_file "$VAE_SRC/$f" "$DEST/analysis/fdvc"
done

# Plotting and visual analysis.
for f in \
  plot_probe_boxplot.py \
  plot_probe_hd95_boxplot.py \
  plot_probe_scatter.py \
  plot_probe_sweep_volume_change_seaborn.py \
  plot_probe_sweep_unified_patient_variability.py \
  plot_probe_sweep_unified_patient_variability_assd.py \
  plot_metric_vs_volume_interval_errorband.py \
  plot_auc_vs_volume_interval_errorband.py \
  visualize_score_distributions.py \
  visualize_score_distributions_final.py \
  visualize_score_distributions_trend.py \
  visualize_score_distributions_trend_levels.py \
  visualize_input_recon_probe_trends.py \
  organize_score_distribution_plots.sh
do
  copy_file "$VAE_SRC/$f" "$DEST/analysis/plots"
done

# Existing shell runners that define the old workflow.
for f in \
  run_all_rois_1p5mm_nozoom_probe_sweep_and_plots.sh \
  run_all_rois_1p5mm_nozoom_probe_sweep_and_plots_assd.sh \
  run_all_rois_1p5mm_nozoom_assd_vis_existing.sh \
  run_roiwise_1p5mm_ae_sweep_one_roi.sh \
  run_lungL_better_probe_sweep.sh \
  run_lungL_better_probe_sweep_v2.sh \
  run_compute_probe_auc_sensitivity.sh \
  run_compute_first_detectable_volume_change.sh \
  run_auc_and_best_detector_with_assd.sh \
  run_select_best_detector_per_roi.sh \
  run_roc_analysis_all_rois_clean.sh \
  run_roc_analysis_lungL_clean.sh \
  run_roc_analysis_lungL_volume_gt20.sh \
  run_roc_analysis_lungL_volume_gt20_ops.sh \
  run_roc_analysis_lungL_volume_gt20_ops_fixed.sh \
  run_analyze_roi_probe_fdvc.sh
do
  copy_file "$VAE_SRC/$f" "$DEST/scripts/cnn_legacy"
done

# Current 1.5-mm no-zoom all-ROI configurations.
copy_dir "$VAE_SRC/configs/roiwise_1p5mm_nozoom_allrois" \
         "$DEST/configs/cnn/roiwise_1p5mm_nozoom_allrois"
for f in label_ae_lungL_roiwise_1p5mm.yaml label_ae_rad_v16.yaml; do
  copy_file "$VAE_SRC/configs/$f" "$DEST/configs/cnn"
done

# Optional Slicer module.
for candidate in \
  "$VAE_SRC/ProbeContourComparer.py" \
  "$DINO_SRC/ProbeContourComparer.py" \
  "/home/shaohan/ProbeContourComparer.py"
do
  if [[ -f "$candidate" ]]; then copy_file "$candidate" "$DEST/slicer"; break; fi
done

if [[ "$DRY_RUN" != "1" ]]; then
  {
    echo "Created: $(date --iso-8601=seconds)"
    echo "DINO source: $DINO_SRC"
    echo "VariantVAE source: $VAE_SRC"
    echo "Destination: $DEST"
    echo
    echo "Copied files:"
    find "$DEST" -path "$DEST/manifests" -prune -o -type f -printf '%P\n' | sort
  } > "$DEST/manifests/copied_files.txt"

  cat > "$DEST/README.md" <<README
# Contour QA Experiments

Organized working copy for comparing the legacy CNN/VariantVAE shape autoencoder
with the frozen pretrained 3DINO shape autoencoder.

Original repositories are unchanged:

- $DINO_SRC
- $VAE_SRC

No pretrained checkpoint, patient dataset, WandB directory, Git history,
model output, or 3DINO CT-reconstruction experiment is copied.
README

  echo "Copy complete: $DEST"
  echo "Manifest: $DEST/manifests/copied_files.txt"
else
  echo "Dry run complete."
fi
