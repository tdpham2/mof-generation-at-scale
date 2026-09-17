#!/bin/bash
# LAMMPS ML-IAP embeds Python and must use its ABI-matched virtual environment.
#
# LAMMPS was built with BUILD_MPI=ON through the Cray compiler wrappers and
# links against cray-mpich's GPU-transport-layer library, so it needs the same
# module stack at runtime that build-lammps.sh used at build time (see that
# script and build-cp2k.sh for why: cray-mpich 9.1.0's libmpi_gtl_cuda.so
# needs libcudart.so.13, and craype-accel-nvidia80 injects -lmpi_gtl_cuda into
# every link). This mirrors run-cp2k-polaris.sh.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

module use /soft/modulefiles
module reset
module use /soft/modulefiles

if module -t list 2>&1 | grep -q '^PrgEnv-nvidia/'; then
    module swap PrgEnv-nvidia PrgEnv-gnu
elif ! module -t list 2>&1 | grep -q '^PrgEnv-gnu/'; then
    module load PrgEnv-gnu
fi
module load gcc-native/14

module unload cray-libsci 2>/dev/null || true
module unload cray-fftw 2>/dev/null || true
module load cray-libsci
module load cray-fftw

module load cuda
module load craype-accel-nvidia80
module unload cuda
module load cudatoolkit-standalone/13.0.1

export CUDA_PATH="${CUDA_HOME}"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MAX_THREAD_SAFETY=multiple

# The site `conda` module's collection has been broken (all its modules were
# removed), so this activates the venv directly instead of going through
# `module load conda` + `conda activate base` first. See build-cp2k.sh's
# local-toolchain note and build-lammps.sh for the build-time equivalent.
lammps_root="${repo_root}/deps/test/lammps-22Jul2025"
source "${lammps_root}/venv/bin/activate"

exec "${lammps_root}/build-mliap-no-mpi/lmp" "$@"
