#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -N mofa-smoke
#PBS -A YOUR_PROJECT

set -euo pipefail

cd "${PBS_O_WORKDIR:?Submit this script from the MOFA repository root}"
repo_root=$PWD
echo "Repository: ${repo_root}"
# Runtime libraries used by the local LAMMPS build.
module reset
module use /soft/modulefiles
module load gcc
module load cudatoolkit-standalone/12.8
module load conda
conda activate base

conda activate "${repo_root}/mofa_env"
export PATH="${repo_root}/mofa_env/bin:${PATH}"
export CP2K_DATA_DIR="${repo_root}/deps/cp2k-2025.1/data"
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Several chemistry/ML imports initialize font and Matplotlib caches. Keep
# them off the home filesystem and give worker processes a writable location.
runtime_cache="${repo_root}/.runtime-cache/${PBS_JOBID}"
mkdir -p "${runtime_cache}/matplotlib" "${runtime_cache}/xdg"
export MPLCONFIGDIR="${runtime_cache}/matplotlib"
export XDG_CACHE_HOME="${runtime_cache}/xdg"

cp2k_shell="${repo_root}/deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.ssmp"
mace_model="${repo_root}/input-files/mace/mace-mp0_medium-mliap_lammps.pt"
lammps_exe="${repo_root}/deps/lammps/build-mliap/lmp"

if [[ ! -x "${cp2k_shell}" ]]; then
    echo "Missing ${cp2k_shell}"
    echo 'Create it with: make -C deps/cp2k-2025.1 ARCH=local_cuda VERSION="ssmp psmp" cp2k_shell'
    exit 2
fi

if [[ ! -f "${mace_model}" ]]; then
    echo "Missing ${mace_model}"
    echo "Prepare it with: (cd input-files/mace && bash get-macemp-0a.sh)"
    exit 2
fi

if [[ ! -x "${lammps_exe}" ]]; then
    echo "Missing ${lammps_exe}"
    echo "Build it first with: qsub polaris-build/build-lammps-polaris.sh"
    exit 2
fi

if ! ldd "${lammps_exe}" | grep -q "${repo_root}/mofa_env/lib/libpython3.10"; then
    echo "LAMMPS is not yet linked to the MOFA Python 3.10 environment."
    echo "Build it first with: qsub polaris-build/build-lammps-polaris.sh"
    exit 2
fi

for required_command in redis-server mongod chargemol simulate monitor_utilization; do
    if ! command -v "${required_command}" >/dev/null; then
        echo "Required command is unavailable: ${required_command}"
        exit 2
    fi
done

redis_log="${repo_root}/redis-${PBS_JOBID}.log"
redis-server \
    --bind 127.0.0.1 \
    --appendonly no \
    --protected-mode no \
    --logfile "${redis_log}" &
redis_pid=$!

cleanup() {
    kill "${redis_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python run_parallel_workflow.py \
    --node-path input-files/zn-paddle-pillar/node.json \
    --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
    --generator-config-path models/geom-300k/config-tf32-a100.yaml \
    --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
    --compute-config configs/polaris/local-smoke.py \
    --mace-model-path "${mace_model}" \
    --redis-host 127.0.0.1 \
    --molecule-sizes 9 12 \
    --num-samples 8 \
    --gen-batch-size 8 \
    --minimum-ligand-pool 2 \
    --simulation-budget 8 \
    --retrain-freq 1000 \
    --num-epochs 1 \
    --md-timesteps 100 \
    --md-snapshots-freq 50 \
    --dft-opt-steps 1 \
    --raspa-timesteps 100
