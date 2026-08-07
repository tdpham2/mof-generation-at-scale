#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N lammps-build
#PBS -A ChemGraph

set -eo pipefail

cd "${PBS_O_WORKDIR:?Submit this script from the MOFA repository root}"
repo_root=$PWD
lammps_src="${repo_root}/deps/lammps"
build_dir="${lammps_src}/build-mliap"

if [[ ! -f "${lammps_src}/cmake/CMakeLists.txt" ]]; then
    echo "Missing LAMMPS source at ${lammps_src}."
    echo "Clone stable_29Aug2024_update3 as described in polaris-build/instruction.md."
    exit 2
fi

if [[ ! -x "${repo_root}/mofa_env/bin/python" ]]; then
    echo "Missing MOFA environment at ${repo_root}/mofa_env."
    echo "Create it from envs/environment-polaris.yml first."
    exit 2
fi

module reset
module use /soft/modulefiles
module load spack-pe-base cmake
module load gcc
module load cudatoolkit-standalone/12.8
module load conda
module list

conda activate base
conda activate "${repo_root}/mofa_env"

export CRAYPE_LINK_TYPE=dynamic
export LDFLAGS="-Wl,--allow-multiple-definition"

cmake \
    -S "${lammps_src}/cmake" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DCMAKE_CXX_COMPILER="${lammps_src}/lib/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_INSTALL_PREFIX="${build_dir}" \
    -DPython_EXECUTABLE="${repo_root}/mofa_env/bin/python" \
    -DPython_ROOT_DIR="${repo_root}/mofa_env" \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_MPI=OFF \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_ZEN3=ON \
    -DKokkos_ARCH_AMPERE80=ON \
    -DPKG_ML-IAP=ON \
    -DPKG_ML-SNAP=ON \
    -DMLIAP_ENABLE_PYTHON=ON \
    -DPKG_PYTHON=ON

cmake --build "${build_dir}" --parallel 16
cmake --install "${build_dir}"
cmake --build "${build_dir}" --target install-python

echo "LAMMPS build completed. Embedded Python library:"
ldd "${build_dir}/lmp" | grep "${repo_root}/mofa_env/lib/libpython3.10"

echo "Enabled packages:"
for package in ML-IAP ML-SNAP KOKKOS PYTHON; do
    "${build_dir}/lmp" -help | grep -qw "${package}"
    echo "  ${package}: enabled"
done

python -c "import lammps; print('LAMMPS Python package:', lammps.__file__)"
