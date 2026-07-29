#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaohan/contour_qa_experiments}"
export PROJECT_ROOT
bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_Lung_L_v1.sh"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_dino_shape_random_morphology_probe_sweep_Lung_L_v1.sh"
bash "${PROJECT_ROOT}/scripts/dino_shape/run_compare_dino_shape_three_models_roc_Lung_L.sh"
