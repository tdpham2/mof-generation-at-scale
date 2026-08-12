#!/bin/bash
# LAMMPS ML-IAP embeds Python and must use its ABI-matched virtual environment.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

module use /soft/modulefiles
module load conda

# Parsl exports the conda function but not all helper functions into this
# child shell. Reload the complete shell integration before changing prefixes.
if [[ -z "${CONDA_EXE:-}" ]]; then
    echo "The conda module did not define CONDA_EXE." >&2
    exit 2
fi
source "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh"
conda activate base

lammps_root="${repo_root}/deps/test/lammps-22Jul2025"
source "${lammps_root}/venv/bin/activate"
module load cudatoolkit-standalone/12.9.1

exec "${lammps_root}/build-mliap-no-mpi/lmp" "$@"
