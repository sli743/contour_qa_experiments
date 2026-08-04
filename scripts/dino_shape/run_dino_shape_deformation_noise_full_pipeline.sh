#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${PROJECT_ROOT}"
for roi in Bronchus Lung_L; do
  bash "scripts/dino_shape/run_preview_dino_shape_deformation_noise_${roi}_v1.sh"
  bash "scripts/dino_shape/run_dino_shape_deformation_noise_${roi}_v1.sh"
  bash "scripts/dino_shape/run_dino_shape_deformation_noise_probe_sweep_${roi}_v1.sh"
  bash "scripts/dino_shape/run_export_dino_shape_deformation_noise_probe_contours_${roi}_v1.sh"
done
bash scripts/dino_shape/run_generate_four_model_reports_Bronchus_Lung_L.sh
