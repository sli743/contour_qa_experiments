#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/shaohan/contour_qa_experiments
CONFIG=${PROJECT_ROOT}/configs/dino_shape/dino_shape_denoising_Lung_L_v1.yaml
cd "${PROJECT_ROOT}"
python -m training.train_dino_shape_denoising_decoder --config "${CONFIG}"
