#!/bin/bash -l

# Run this script from an active Polaris PBS allocation.
set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Run this script inside a Polaris PBS job."
    exit 2
fi

test_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd "${test_dir}/.." && pwd -P)
affinity_script="${repo_root}/bin/set-affinity-gpu-polaris.sh"
cp2k_wrapper="${repo_root}/bin/run-cp2k-polaris.sh"

if [[ ! -x "${cp2k_wrapper}" ]]; then
    echo "Missing CP2K wrapper: ${cp2k_wrapper}"
    exit 2
fi
if [[ ! -f "${test_dir}/cp2k.inp" ]]; then
    echo "Missing CP2K input: ${test_dir}/cp2k.inp"
    exit 2
fi

export OMP_NUM_THREADS=8

run_dir="${test_dir}/run-${PBS_JOBID}"
if [[ -e "${run_dir}" ]]; then
    echo "Run directory already exists: ${run_dir}"
    exit 2
fi
mkdir "${run_dir}"
cp "${test_dir}/cp2k.inp" "${run_dir}/"
cd "${run_dir}"

echo "CP2K wrapper: ${cp2k_wrapper}"
echo "Run directory: ${run_dir}"

# bin/run-cp2k-polaris.sh loads the CP2K runtime module stack itself (see
# build-cp2k.sh) and execs cp2k_shell.psmp, so this script does not need to
# duplicate that setup.
env MPICH_OFI_CXI_PID_BASE=5 \
    mpiexec -n 4 --ppn 4 --cpu-bind depth --depth 8 \
    -env OMP_NUM_THREADS=8 -env CP2K_BINARY=cp2k.psmp \
    "${affinity_script}" "${cp2k_wrapper}" \
    -i cp2k.inp -o cp2k.out \
    > launcher.stdout 2> launcher.stderr

if ! grep -q 'PROGRAM ENDED AT' cp2k.out; then
    echo "CP2K exited without its successful-termination marker."
    tail -n 100 cp2k.out
    exit 3
fi

echo "CP2K simulation completed successfully: ${run_dir}/cp2k.out"
