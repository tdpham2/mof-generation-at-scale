#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N cp2k-build
#PBS -A ChemGraph
#PBS -j oe

# Submit with:
#   qsub /lus/eagle/projects/ChemGraph/thang/soft/cp2k-2025.1/build-cp2k.sh
set -euo pipefail

if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "Submit this script with qsub; it must run inside a Polaris PBS job."
    exit 2
fi

cp2k_root="/lus/eagle/projects/ChemGraph/thang/soft/cp2k-2025.1"
build_jobs=8

if [[ ! -f "${cp2k_root}/CMakeLists.txt" ]]; then
    echo "Missing CP2K source at ${cp2k_root}."
    exit 2
fi
if ! grep -q 'VERSION "2025.1"' "${cp2k_root}/CMakeLists.txt"; then
    echo "The source at ${cp2k_root} is not CP2K 2025.1."
    exit 2
fi

# Prevent concurrent jobs from moving or rebuilding the same directories.
lock_file="${cp2k_root}/.build-cp2k.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
    echo "ERROR: Another CP2K build is active for ${cp2k_root}."
    exit 3
fi

cd "${cp2k_root}"
echo "PBS job: ${PBS_JOBID}"
echo "CP2K source: ${cp2k_root}"

echo "========== Configuring Polaris modules =========="
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

# The CPE accelerator module provides the A100 target and CUDA-aware Cray MPI
# linkage, but it does not recognize the site standalone CUDA module. Satisfy
# its prerequisite with CPE CUDA 11.8, retain the accelerator target, and then
# select CUDA 12.8.1 for the actual toolchain and runtime libraries.
module load cuda/11.8
module load craype-accel-nvidia80
module unload cuda/11.8
module load cudatoolkit-standalone/12.8.1

export CUDA_PATH="${CUDA_HOME}"
export MPICH_GPU_SUPPORT_ENABLED=1

module list

echo
echo "========== Compiler environment =========="
echo "PE_ENV=${PE_ENV:-}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CRAY_ACCEL_TARGET=${CRAY_ACCEL_TARGET:-}"
cc --version | head -n 1
CC --version | head -n 1
ftn --version | head -n 1
nvcc --version | tail -n 1

if ! nvcc --version | grep -q 'release 12\.8'; then
    echo "ERROR: CP2K must be built with CUDA 12.8.x on Polaris."
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

# COSMA and COSTA require the LibSci installation prefix. Derive the active
# GNU/GCC 12.3 path from the Cray compiler wrapper instead of hard-coding it.
libsci_libdir=$(
    cc --cray-print-opts=libs |
        tr ' ' '\n' |
        sed -n 's/^-L//p' |
        grep '/libsci/' |
        head -n 1 || true
)
if [[ -z "${libsci_libdir}" || ! -d "${libsci_libdir}" ]]; then
    echo "ERROR: Could not determine the active Cray LibSci directory."
    exit 1
fi
case "${libsci_libdir}" in
    */lib|*/lib64)
        export CRAY_LIBSCI_PREFIX_DIR="${libsci_libdir%/*}"
        ;;
    *)
        echo "ERROR: Unexpected LibSci library path: ${libsci_libdir}"
        exit 1
        ;;
esac
export CRAY_PE_LIBSCI_PREFIX_DIR="${CRAY_LIBSCI_PREFIX_DIR}"
echo "CRAY_LIBSCI_PREFIX_DIR=${CRAY_LIBSCI_PREFIX_DIR}"

echo
echo "========== Preserving previous build =========="
backup_tag="pre-cuda128-$(date -u +%Y%m%dT%H%M%SZ)"
for old_path in \
    tools/toolchain/install \
    tools/toolchain/build \
    obj/local \
    lib/local \
    exe/local \
    obj/local_cuda \
    lib/local_cuda \
    exe/local_cuda
do
    if [[ -e "${old_path}" ]]; then
        echo "Moving ${old_path} to ${old_path}.${backup_tag}"
        mv "${old_path}" "${old_path}.${backup_tag}"
    fi
done
if [[ -f tools/toolchain/install.log ]]; then
    mv tools/toolchain/install.log \
        "tools/toolchain/install.log.${backup_tag}"
fi

echo
echo "========== Installing CP2K dependencies =========="
cd tools/toolchain
./install_cp2k_toolchain.sh \
    --gpu-ver=A100 \
    --enable-cuda \
    --target-cpu=znver3 \
    --mpi-mode=mpich \
    --with-elpa=install \
    --with-sirius=no \
    -j "${build_jobs}" 2>&1 | tee install.log

cp install/arch/* ../../arch/
cd "${cp2k_root}"

echo
echo "========== Building CP2K =========="
# The generated toolchain setup appends to CP_* variables without first
# defining them. Temporarily disable nounset while importing it, then restore
# the build driver's strict shell settings.
set +u
source ./tools/toolchain/install/setup
set -u
export MPICH_GPU_SUPPORT_ENABLED=1

if ! nvcc --version | grep -q 'release 12\.8'; then
    echo "ERROR: The generated toolchain changed the active CUDA version."
    nvcc --version
    exit 1
fi

make -j "${build_jobs}" ARCH=local VERSION="ssmp psmp"
make -j "${build_jobs}" ARCH=local_cuda VERSION="ssmp psmp"

echo
echo "========== Verifying executables =========="
executables=(
    exe/local/cp2k.ssmp
    exe/local/cp2k.psmp
    exe/local/cp2k_shell.ssmp
    exe/local/cp2k_shell.psmp
    exe/local_cuda/cp2k.ssmp
    exe/local_cuda/cp2k.psmp
    exe/local_cuda/cp2k_shell.ssmp
    exe/local_cuda/cp2k_shell.psmp
)
for exe in "${executables[@]}"; do
    if [[ ! -x "${exe}" ]]; then
        echo "ERROR: Missing executable: ${exe}"
        exit 1
    fi
    echo "--- ${exe} ---"
    "./${exe}" --version | sed -n '1,8p'
done

echo
echo "========== Verifying CUDA runtime and MPI GTL =========="
for exe in exe/local_cuda/cp2k.psmp exe/local_cuda/cp2k_shell.psmp; do
    dynamic_section=$(readelf -d "${exe}")
    if [[ "${dynamic_section}" != *"cuda-12.8.1"* ]]; then
        echo "ERROR: ${exe} does not contain the CUDA 12.8.1 runpath."
        exit 1
    fi
    if [[ "${dynamic_section}" == *"cuda-12.9"* ]]; then
        echo "ERROR: ${exe} still contains a CUDA 12.9 runpath."
        exit 1
    fi

    ldd_output=$(ldd "${exe}")
    if [[ "${ldd_output}" == *"not found"* ]]; then
        echo "ERROR: ${exe} has unresolved shared libraries:"
        echo "${ldd_output}"
        exit 1
    fi
    if [[ "${ldd_output}" != *"libmpi_gtl_cuda"* ]]; then
        echo "ERROR: ${exe} is not linked to the Cray MPI CUDA GTL."
        exit 1
    fi
    if ! grep 'libnvrtc' <<<"${ldd_output}" | grep -q 'cuda-12.8.1'; then
        echo "ERROR: ${exe} does not resolve NVRTC from CUDA 12.8.1."
        echo "${ldd_output}" | grep -E 'cuda|nvrtc|gtl' || true
        exit 1
    fi

    libsci_count=0
    while IFS= read -r libsci_path; do
        [[ -n "${libsci_path}" ]] || continue
        libsci_count=$((libsci_count + 1))
        resolved_libsci=$(readlink -f "${libsci_path}")
        case "${resolved_libsci}" in
            "${CRAY_LIBSCI_PREFIX_DIR}"/*) ;;
            *)
                echo "ERROR: ${exe} resolves LibSci outside the active installation:"
                echo "  ${resolved_libsci}"
                exit 1
                ;;
        esac
    done < <(awk '/libsci/ {print $3}' <<<"${ldd_output}")
    if (( libsci_count == 0 )); then
        echo "ERROR: ${exe} has no resolved Cray LibSci dependency."
        exit 1
    fi

    echo "--- ${exe}: CUDA, NVRTC, MPI GTL, and LibSci ---"
    echo "${ldd_output}" | grep -Ei 'cuda|nvrtc|gtl|libsci'
done

echo
echo "CP2K installation completed successfully."
echo "Previous build suffix: ${backup_tag}"
