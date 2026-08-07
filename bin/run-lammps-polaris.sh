#!/bin/bash
# LAMMPS ML-IAP embeds Python and must use the same environment as MOFA/MACE.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
module use /soft/modulefiles; module load conda; conda activate base

conda activate "${repo_root}/mofa_env"
exec "${repo_root}/deps/lammps/build-mliap/lmp" "$@"
