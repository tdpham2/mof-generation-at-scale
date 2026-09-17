#!/bin/bash -l

# Build LAMMPS (stable_22Jul2025) with Kokkos/CUDA and ML-IAP for MOFA on
# Polaris. Run this from an active Polaris PBS compute-node allocation, from the
# MOFA repository root, e.g.:
#   qsub -I -l select=1:system=polaris -l walltime=01:00:00 \
#        -l filesystems=home:eagle -q debug -A ChemGraph
#   bash polaris-build/build-lammps.sh
#
# LAMMPS ML-IAP embeds Python and PyTorch, so it is built against its own venv
# (Python 3.12) inside the build tree, NOT the MOFA conda environment. The
# runtime wrapper bin/run-lammps-polaris.sh activates this same venv.
#
# The module stack below mirrors build-cp2k.sh. LAMMPS is built with
# BUILD_MPI=ON through the Cray compiler wrappers (cc/CC via nvcc_wrapper), so
# it is subject to the same constraint documented there: cray-mpich 9.1.0's
# libmpi_gtl_cuda.so needs libcudart.so.13, and craype-accel-nvidia80 injects
# -lmpi_gtl_cuda into every link. Polaris's GPUs are A100s (nvidia80), so the
# accelerator target must match too.
set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Run this script inside a Polaris PBS compute-node job."
    exit 2
fi

#repo_root="${PBS_O_WORKDIR:-$PWD}"
repo_root="/lus/eagle/projects/datascience/hari/mof-generation-at-scale"
lammps_root="${repo_root}/deps/test/lammps-22Jul2025"
build_dir="${lammps_root}/build-mliap-no-mpi"

if [[ ! -f "${lammps_root}/cmake/CMakeLists.txt" ]]; then
    echo "Missing LAMMPS source at ${lammps_root}."
    echo "Clone stable_22Jul2025_update5 as described in polaris-build/instruction.md."
    exit 2
fi

echo "========== Configuring Polaris modules =========="
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

# See build-cp2k.sh for why this must be CUDA 13.0.x: cray-mpich/9.1.0's
# libmpi_gtl_cuda.so has "NEEDED libcudart.so.13", and craype-accel-nvidia80
# injects -lmpi_gtl_cuda into every link.
module load cuda
module load craype-accel-nvidia80
module unload cuda
module load cudatoolkit-standalone/13.0.1

export CUDA_PATH="${CUDA_HOME}"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_MAX_THREAD_SAFETY=multiple

export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LDFLAGS="-L${CUDA_HOME}/lib64 -lcudart ${LDFLAGS:-}"

module unload darshan

module load spack-pe-base cmake

# The site `conda` module's collection has been broken (all its modules were
# removed) -- see build-cp2k.sh's local toolchain note. Use the local miniconda
# from ~/.bashrc instead, with a dedicated Python 3.12 env (matching the
# Python 3.12 the site `conda` module used to provide) so the LAMMPS venv gets
# torch==2.5.0 wheels (no 3.14 build exists, which is what miniconda's own
# `base` env now carries).
miniconda_root="/lus/eagle/projects/datascience/hari/.local/miniconda3"
if [[ ! -f "${miniconda_root}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: Missing local miniconda at ${miniconda_root}."
    exit 2
fi
source "${miniconda_root}/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^lammps-py312 '; then
    echo "ERROR: Missing local conda env 'lammps-py312' (create with:"
    echo "       conda create -n lammps-py312 python=3.12)."
    exit 2
fi
conda activate lammps-py312
module list

echo
echo "========== Compiler environment =========="
echo "PE_ENV=${PE_ENV:-}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CRAY_ACCEL_TARGET=${CRAY_ACCEL_TARGET:-}"
cc --version | head -n 1
CC --version | head -n 1
nvcc --version | tail -n 1

if ! nvcc --version | grep -q 'release 13\.0'; then
    echo "ERROR: LAMMPS must be built with CUDA 13.0.x on Polaris"
    echo "       (cray-mpich 9.1.0 GTL requires libcudart.so.13)."
    nvcc --version
    exit 1
fi
if [[ "${CRAY_ACCEL_TARGET:-}" != "nvidia80" ]]; then
    echo "ERROR: Expected the Polaris A100 target nvidia80."
    exit 1
fi
if ! cc --cray-print-opts=libs | grep -q -- '-lmpi_gtl_cuda'; then
    echo "ERROR: The Cray compiler wrapper is not configured for CUDA-aware MPI."
    echo "cc --cray-print-opts=libs returned:"
    cc --cray-print-opts=libs
    exit 1
fi

cd "${lammps_root}"

echo
echo "========== Creating the LAMMPS venv =========="
pwd
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r ./python/wheel_requirements.txt
python3 -m pip install \
    "torch==2.5.0" \
    cuequivariance-torch \
    cuequivariance \
    cuequivariance-ops-torch-cu12 \
    cupy-cuda12x \
    "mace-torch==0.3.13" \
    cython \
    "numpy<2"

echo
echo "========== Building LAMMPS =========="
export NVCC_WRAPPER_DEFAULT_COMPILER=CC

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${build_dir}" \
    -D BUILD_MPI=ON \
    -D BUILD_SHARED_LIBS=ON \
    \
    -D CMAKE_C_COMPILER=cc \
    -D CMAKE_CXX_COMPILER="${lammps_root}/lib/kokkos/bin/nvcc_wrapper" \
    \
    -D PKG_KOKKOS=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_EXTRA-MOLECULE=ON \
    -D PKG_KSPACE=ON \
    -D PKG_ML-SNAP=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_PYTHON=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    \
    -D Kokkos_ENABLE_CUDA=ON \
    -D FFT_KOKKOS=CUFFT \
    -D FFT_SINGLE=yes \
    -D Kokkos_ENABLE_OPENMP=ON \
    \
    -D CMAKE_EXE_LINKER_FLAGS="-target-accel=nvidia80" \
    -D Kokkos_ARCH_AMDAVX=ON \
    -S "${lammps_root}/cmake" \
    -B "${build_dir}"

cmake --build "${build_dir}" --parallel 32
cmake --build "${build_dir}" --target install-python

echo
echo "========== Verifying the build =========="
if [[ ! -x "${build_dir}/lmp" ]]; then
    echo "ERROR: Missing LAMMPS executable: ${build_dir}/lmp"
    exit 1
fi

for package in ML-IAP ML-SNAP KOKKOS PYTHON EXTRA-MOLECULE; do
    "${build_dir}/lmp" -help | grep -qw "${package}"
    echo "  ${package}: enabled"
done

python -c "import lammps; print('LAMMPS Python package:', lammps.__file__)"

echo
echo "LAMMPS installation completed successfully: ${build_dir}/lmp"
echo "Next: prepare the MACE model (see polaris-build/instruction.md step 5)."
