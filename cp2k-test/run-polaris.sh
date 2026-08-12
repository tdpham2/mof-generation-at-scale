#!/bin/bash -l

# Run this script from an active Polaris PBS allocation.
set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Run this script inside a Polaris PBS job."
    exit 2
fi

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${test_dir}/.." && pwd -P)
cp2k_exe="${repo_root}/deps/cp2k-2025.1/exe/local_cuda/cp2k.psmp"
cp2k_data="${repo_root}/deps/cp2k-2025.1/data"
affinity_script="${repo_root}/bin/set-affinity-gpu-polaris.sh"

if [[ ! -x "${cp2k_exe}" ]]; then
    echo "Missing CP2K executable: ${cp2k_exe}"
    exit 2
fi
if [[ ! -f "${test_dir}/cp2k.inp" ]]; then
    echo "Missing CP2K input: ${test_dir}/cp2k.inp"
    exit 2
fi

# Match the CP2K runtime environment in configs/polaris/polaris-repo.py.
module reset
module use /soft/modulefiles
if module -t list 2>&1 | grep -q '^PrgEnv-nvidia/'; then
    module swap PrgEnv-nvidia PrgEnv-gnu
elif ! module -t list 2>&1 | grep -q '^PrgEnv-gnu/'; then
    module load PrgEnv-gnu
fi
if module -t list 2>&1 | grep -q '^gcc-native/14'; then
    module swap gcc-native/14 gcc-native/12.3
else
    module load gcc-native/12.3
fi

module unload cray-libsci 2>/dev/null || true
module unload cray-fftw 2>/dev/null || true
module load cray-libsci
module load cray-fftw
module load cuda/11.8
module load craype-accel-nvidia80
module unload cuda/11.8
module load cudatoolkit-standalone/12.8.1

export CUDA_PATH="${CUDA_HOME}"
export MPICH_GPU_SUPPORT_ENABLED=1
export CP2K_DATA_DIR="${cp2k_data}"
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=8

run_dir="${test_dir}/run-${PBS_JOBID}"
if [[ -e "${run_dir}" ]]; then
    echo "Run directory already exists: ${run_dir}"
    exit 2
fi
mkdir "${run_dir}"
cp "${test_dir}/cp2k.inp" "${run_dir}/"
cd "${run_dir}"

echo "CP2K executable: ${cp2k_exe}"
echo "Run directory: ${run_dir}"

env MPICH_OFI_CXI_PID_BASE=5 \
    mpiexec -n 4 --ppn 4 --cpu-bind depth --depth 8 \
    -env OMP_NUM_THREADS=8 \
    "${affinity_script}" "${cp2k_exe}" \
    -i cp2k.inp -o cp2k.out \
    > launcher.stdout 2> launcher.stderr

if ! grep -q 'PROGRAM ENDED AT' cp2k.out; then
    echo "CP2K exited without its successful-termination marker."
    tail -n 100 cp2k.out
    exit 3
fi

echo "CP2K simulation completed successfully: ${run_dir}/cp2k.out"
