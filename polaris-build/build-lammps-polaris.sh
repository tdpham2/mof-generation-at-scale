#!/bin/bash -l

# Build LAMMPS (stable_22Jul2025) with Kokkos/CUDA and ML-IAP for MOFA on
# Polaris. Run this from an active Polaris PBS compute-node allocation, from the
# MOFA repository root, e.g.:
#   qsub -I -l select=1:system=polaris -l walltime=01:00:00 \
#        -l filesystems=home:eagle -q debug -A ChemGraph
#   bash polaris-build/build-lammps-polaris.sh
#
# LAMMPS ML-IAP embeds Python and PyTorch, so it is built against its own venv
# (Python 3.12) inside the build tree, NOT the MOFA conda environment. The
# runtime wrapper bin/run-lammps-polaris.sh activates this same venv.
set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Run this script inside a Polaris PBS compute-node job."
    exit 2
fi

repo_root="${PBS_O_WORKDIR:-$PWD}"
lammps_root="${repo_root}/deps/test/lammps-22Jul2025"
build_dir="${lammps_root}/build-mliap-no-mpi"

if [[ ! -f "${lammps_root}/cmake/CMakeLists.txt" ]]; then
    echo "Missing LAMMPS source at ${lammps_root}."
    echo "Clone stable_22Jul2025_update5 as described in polaris-build/instruction.md."
    exit 2
fi

echo "========== Configuring Polaris modules =========="
module restore
module swap PrgEnv-nvidia PrgEnv-gnu

# Provide the A100 target and CUDA-aware Cray MPI linkage. cuda is loaded only
# to satisfy the accelerator module's prerequisite, then unloaded in favor of
# the standalone CUDA toolkit used for the actual build.
module load cuda
module load craype-accel-nvidia90
module unload cuda

module use /soft/modulefiles
module load cudatoolkit-standalone/12.9.1

module load spack-pe-base cmake
module load cray-fftw

module load conda
conda activate base
module list

cd "${lammps_root}"

echo
echo "========== Creating the LAMMPS venv =========="
python3 -m venv venv
source venv/bin/activate
pip install -r ../python/wheel_requirements.txt

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
    -D CMAKE_EXE_LINKER_FLAGS="-target-accel=nvidia90" \
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

for package in ML-IAP ML-SNAP KOKKOS PYTHON; do
    "${build_dir}/lmp" -help | grep -qw "${package}"
    echo "  ${package}: enabled"
done

python -c "import lammps; print('LAMMPS Python package:', lammps.__file__)"

echo
echo "LAMMPS installation completed successfully: ${build_dir}/lmp"
echo "Next: install MACE into this venv (see polaris-build/instruction.md step 4)."
