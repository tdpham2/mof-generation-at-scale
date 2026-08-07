#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:eagle
#PBS -q capacity
#PBS -N cp2k-build
#PBS -A YOUR_PROJECT

set -eo pipefail

cd "${PBS_O_WORKDIR:?Submit this script from the MOFA repository root}"
repo_root=$PWD
cp2k_root="${repo_root}/deps/cp2k-2025.1"

if [[ ! -f "${cp2k_root}/CMakeLists.txt" ]]; then
    echo "Missing CP2K 2025.1 source at ${cp2k_root}."
    echo "Clone v2025.1 as described in polaris-build/instruction.md."
    exit 2
fi

if ! grep -q 'VERSION "2025.1"' "${cp2k_root}/CMakeLists.txt"; then
    echo "The source at ${cp2k_root} is not CP2K 2025.1."
    exit 2
fi

cd "${cp2k_root}"

echo "========== Configuring Polaris modules =========="

module reset
module use /soft/modulefiles

# Use GNU compilers through the Cray compiler wrappers.
if module -t list 2>&1 | grep -q '^PrgEnv-nvidia/'; then
    module swap PrgEnv-nvidia PrgEnv-gnu
elif ! module -t list 2>&1 | grep -q '^PrgEnv-gnu/'; then
    module load PrgEnv-gnu
fi

# Keep NVIDIA/CUDA support available while using PrgEnv-gnu.
module load nvidia-mixed

# CUDA 12.2 requires GCC 12 rather than the current GCC 14 default.
if module -t list 2>&1 | grep -q '^gcc-native/14'; then
    module swap gcc-native/14 gcc-native/12.3
else
    module load gcc-native/12.3
fi

# Reload compiler-dependent libraries after changing the GCC version.
module unload cray-libsci 2>/dev/null || true
module unload cray-fftw 2>/dev/null || true
module load cray-libsci
module load cray-fftw
module load cudatoolkit-standalone/12.2.2
module list

echo
echo "========== Compiler environment =========="
echo "PE_ENV=${PE_ENV:-}"
cc --version | head -n 1
CC --version | head -n 1
ftn --version | head -n 1
nvcc --version | tail -n 1

# COSMA/COSTA expect CRAY_LIBSCI_PREFIX_DIR, but current Polaris modules may
# not populate it. Derive the LibSci path from the Cray compiler wrapper.
LIBSCI_LIBDIR="$(
    cc --cray-print-opts=libs |
        tr ' ' '\n' |
        sed -n 's/^-L//p' |
        grep '/libsci/' |
        head -n 1 || true
)"

if [[ -z "${LIBSCI_LIBDIR}" || ! -d "${LIBSCI_LIBDIR}" ]]; then
    echo "ERROR: Could not determine the Cray LibSci library directory."
    echo "cc --cray-print-opts=libs returned:"
    cc --cray-print-opts=libs
    exit 1
fi

case "${LIBSCI_LIBDIR}" in
    */lib|*/lib64)
        export CRAY_LIBSCI_PREFIX_DIR="${LIBSCI_LIBDIR%/*}"
        ;;
    *)
        echo "ERROR: Unexpected LibSci library path: ${LIBSCI_LIBDIR}"
        exit 1
        ;;
esac

export CRAY_PE_LIBSCI_PREFIX_DIR="${CRAY_LIBSCI_PREFIX_DIR}"

echo "CRAY_LIBSCI_PREFIX_DIR=${CRAY_LIBSCI_PREFIX_DIR}"
echo "NVIDIA_PATH=${NVIDIA_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

if ! find "${LIBSCI_LIBDIR}" -maxdepth 1 \
    \( -name 'libsci*.so*' -o -name 'libsci*.a' \) \
    -print -quit | grep -q .; then
    echo "ERROR: No LibSci library was found in ${LIBSCI_LIBDIR}."
    exit 1
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
    -j 8 2>&1 | tee install.log

cp install/arch/* ../../arch/
cd "${cp2k_root}"

echo
echo "========== Building CP2K =========="

source ./tools/toolchain/install/setup
make -j 8 ARCH=local VERSION="ssmp psmp"
make -j 8 ARCH=local_cuda VERSION="ssmp psmp"

echo
echo "========== Verifying executables =========="

for exe in \
    exe/local/cp2k.ssmp \
    exe/local/cp2k.psmp \
    exe/local/cp2k_shell.ssmp \
    exe/local/cp2k_shell.psmp \
    exe/local_cuda/cp2k.ssmp \
    exe/local_cuda/cp2k.psmp \
    exe/local_cuda/cp2k_shell.ssmp \
    exe/local_cuda/cp2k_shell.psmp
do
    if [[ ! -x "${exe}" ]]; then
        echo "ERROR: Missing executable: ${exe}"
        exit 1
    fi

    echo
    echo "--- ${exe} ---"
    "./${exe}" --version | sed -n '1,8p'
done

echo
echo "CP2K installation completed successfully."
