#!/bin/bash -l
#PBS -l select=8:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -N mofa-test
#PBS -A ChemGraph

set -euo pipefail

cd "${PBS_O_WORKDIR:?Submit this script from the MOFA repository root}"
repo_root=$PWD
echo "Repository: ${repo_root}"

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
# Keep Python multiprocessing sockets below the AF_UNIX path-length limit.
# LocalProvider workers inherit this node-local temporary directory.
export TMPDIR=/tmp

runtime_cache="${repo_root}/.runtime-cache/${PBS_JOBID}"
mkdir -p "${runtime_cache}/matplotlib" "${runtime_cache}/xdg"
export MPLCONFIGDIR="${runtime_cache}/matplotlib"
export XDG_CACHE_HOME="${runtime_cache}/xdg"

lammps_root="${repo_root}/deps/test/lammps-22Jul2025"
lammps_exe="${lammps_root}/build-mliap-no-mpi/lmp"
lammps_activate="${lammps_root}/venv/bin/activate"
mace_model="${repo_root}/input-files/mace/mace-mp0_medium-mliap_lammps.pt"
affinity_script="${repo_root}/bin/set-affinity-gpu-polaris.sh"


redis_major=$(python -c 'import redis; print(redis.__version__.split(".")[0])')
if [[ "${redis_major}" != "5" ]]; then
    echo "MOFA requires redis-py 5.x for Colmena 0.7; found $(python -c 'import redis; print(redis.__version__)')"
    exit 2
fi

num_nodes=$(wc -l < "${PBS_NODEFILE}")
echo "Allocated nodes: ${num_nodes}"

redis_pid=""
mps_started=false
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${redis_pid}" ]]; then
        kill "${redis_pid}" 2>/dev/null || true
        wait "${redis_pid}" 2>/dev/null || true
    fi
    if [[ "${mps_started}" == true ]]; then
        mpiexec --no-vni -n "${num_nodes}" --ppn 1 "${repo_root}/bin/disable_mps_polaris.sh" || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

mpiexec --no-vni -n "${num_nodes}" --ppn 1 "${repo_root}/bin/enable_mps_polaris.sh"
mps_started=true

redis_host=$(hostname)
redis_log="${repo_root}/redis-${PBS_JOBID}.log"
redis-server \
    --bind 0.0.0.0 \
    --appendonly no \
    --maxclients 1000000 \
    --protected-mode no \
    --logfile "${redis_log}" &
redis_pid=$!

for _ in $(seq 1 30); do
    if redis-cli -h "${redis_host}" ping 2>/dev/null | grep -q PONG; then
        break
    fi
    if ! kill -0 "${redis_pid}" 2>/dev/null; then
        echo "Redis exited during startup. See ${redis_log}."
        exit 2
    fi
    sleep 1
done
if ! redis-cli -h "${redis_host}" ping 2>/dev/null | grep -q PONG; then
    echo "Redis did not become ready. See ${redis_log}."
    exit 2
fi

python run_parallel_workflow.py \
    --node-path input-files/zn-paddle-pillar/node.json \
    --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
    --generator-config-path models/geom-300k/config-tf32-a100.yaml \
    --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
    --ai-fraction 0.25 \
    --dft-fraction 0.1 \
    --retrain-freq 2 \
    --num-epochs 4 \
    --num-samples 128 \
    --gen-batch-size 64 \
    --simulation-budget -1 \
    --compute-config configs/polaris/polaris-repo.py \
    --mace-model-path "${mace_model}" \
    --redis-host "${redis_host}" \
    --md-timesteps 1000 \
    --proxy-threshold 1000 \
    --raspa-timesteps 10000 \
    --dft-opt-steps 2 \
    --lammps-on-ramdisk
