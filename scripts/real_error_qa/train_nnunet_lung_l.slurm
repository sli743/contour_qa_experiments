#!/bin/bash
#SBATCH -p hugo
#SBATCH --account=geoffrey_hugo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaohan@wustl.edu
#SBATCH --output=slurm-nnunet-lungl-%A_%a.out

set -euo pipefail

# Submit with:
#   sbatch --array=0-4 scripts/real_error_qa/train_nnunet_lung_l.slurm
#
# Optional overrides at submission time, for example:
#   sbatch --array=0-4 --export=ALL,NNUNET_ENV=nnunet,NNUNET_DATASET_ID=501 \
#     scripts/real_error_qa/train_nnunet_lung_l.slurm

NNUNET_ENV="${NNUNET_ENV:-nnunet}"
NNUNET_DATASET_ID="${NNUNET_DATASET_ID:-501}"
NNUNET_CONFIGURATION="${NNUNET_CONFIGURATION:-3d_fullres}"
NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainer}"
NNUNET_PLANS="${NNUNET_PLANS:-nnUNetPlans}"

export nnUNet_raw="${nnUNet_raw:-/scratch/shaohan/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/scratch/shaohan/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/scratch/shaohan/nnUNet_results}"

if [[ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]]; then
    # This matches the user's existing /home/shaohan/miniconda installation.
    source "$HOME/miniconda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: could not find conda.sh under ~/miniconda or ~/miniconda3" >&2
    exit 2
fi

conda activate "$NNUNET_ENV"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: submit this script as a Slurm array, e.g. --array=0-4" >&2
    exit 2
fi

FOLD="$SLURM_ARRAY_TASK_ID"
if (( FOLD < 0 || FOLD > 4 )); then
    echo "ERROR: expected fold 0..4, got ${FOLD}" >&2
    exit 2
fi

command -v nnUNetv2_train >/dev/null 2>&1 || {
    echo "ERROR: nnUNetv2_train not found in conda environment '${NNUNET_ENV}'" >&2
    exit 2
}

echo "Host:                  $(hostname)"
echo "Date:                  $(date -Is)"
echo "Conda env:             ${NNUNET_ENV}"
echo "Dataset ID:            ${NNUNET_DATASET_ID}"
echo "Configuration:         ${NNUNET_CONFIGURATION}"
echo "Trainer:               ${NNUNET_TRAINER}"
echo "Plans:                 ${NNUNET_PLANS}"
echo "Fold:                  ${FOLD}"
echo "nnUNet_raw:            ${nnUNet_raw}"
echo "nnUNet_preprocessed:   ${nnUNet_preprocessed}"
echo "nnUNet_results:        ${nnUNet_results}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# --npz keeps validation probabilities. The hard validation segmentations are
# written automatically by nnU-Net and will later be collected as OOF outputs.
nnUNetv2_train \
    "$NNUNET_DATASET_ID" \
    "$NNUNET_CONFIGURATION" \
    "$FOLD" \
    -tr "$NNUNET_TRAINER" \
    -p "$NNUNET_PLANS" \
    --npz
