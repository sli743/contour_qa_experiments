#!/usr/bin/env bash
set -euo pipefail

# PyTorch 2.0 used by the 3DINO environment does not recognize
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
unset PYTORCH_CUDA_ALLOC_CONF

python -m analysis.audit_v2_loss_direction "$@"
