#!/usr/bin/env bash
# Shared environment activation for all-ROI jobs.
# Do not rely on the shell that invoked sbatch having conda on PATH.

CONDA_ENV="${CONDA_ENV:-3DINO}"
CONDA_BASE="${CONDA_BASE:-/home/shaohan/miniconda}"

_conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "${_conda_sh}" ]]; then
  # Conservative fallbacks for interactive/non-cluster use.
  for candidate in "${HOME:-}/miniconda" "${HOME:-}/miniconda3"; do
    [[ -n "${candidate}" ]] || continue
    if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
      CONDA_BASE="${candidate}"
      _conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if [[ ! -f "${_conda_sh}" ]]; then
  echo "Could not find conda activation script." >&2
  echo "Tried CONDA_BASE=${CONDA_BASE} and common locations under HOME=${HOME:-<unset>}." >&2
  echo "Set CONDA_BASE to the Miniconda/Anaconda installation root before running." >&2
  return 127 2>/dev/null || exit 127
fi

# shellcheck disable=SC1090
source "${_conda_sh}"
conda activate "${CONDA_ENV}"

if ! command -v python >/dev/null 2>&1; then
  echo "Python is not available after activating conda environment ${CONDA_ENV}." >&2
  return 127 2>/dev/null || exit 127
fi

echo "Activated conda environment: ${CONDA_ENV}"
echo "Conda base: ${CONDA_BASE}"
echo "Python: $(command -v python)"
