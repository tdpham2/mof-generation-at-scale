# Installing and Running MOFA on Polaris

This guide creates a repository-local MOFA installation on Polaris. It reflects
the setup currently proven to run the workflow end to end. CP2K is built under
`deps/`, LAMMPS under `deps/test/`, and MOFA itself lives in a Conda environment
at `./mofa_env`.

Run all commands from the MOFA repository root unless a step says otherwise.

> **Two independent Python environments.** MOFA, Redis, MongoDB, RASPA2, MACE,
> and the steering workflow use the Conda environment `./mofa_env` (Python 3.10).
> LAMMPS is built and run against a **separate venv** inside its own build tree
> (Python 3.12). This is deliberate — see [Executor
> environments](#executor-environments). Do not try to run LAMMPS from
> `mofa_env`.

## Prerequisites

You need:

- A Polaris account and a project allocation.
- A clone of this repository on a filesystem visible to the compute nodes
  (Eagle or Flare).
- Network access from the login node to fetch CP2K, LAMMPS, and the MACE model.

The PBS run scripts set `#PBS -A ChemGraph`. Replace `ChemGraph` with your own
Polaris allocation in these files before submitting:

```text
run-polaris-local-smoke.sh
run-polaris-repo-test.sh
```

`build-cp2k.sh` runs inside an interactive PBS job (see below), so it has no
`#PBS -A` line to edit. The LAMMPS build
(`polaris-build/build-lammps-polaris.sh`) also runs interactively.

## 1. Download the CP2K and LAMMPS sources

Use the versions validated by this repository: **CP2K 2025.1** and **LAMMPS
`stable_22Jul2025` (update5)**.

```bash
mkdir -p deps deps/test

# CP2K 2025.1 -> deps/cp2k-2025.1
git clone --branch v2025.1 --depth 1 \
  https://github.com/cp2k/cp2k.git deps/cp2k-2025.1

# LAMMPS 22Jul2025 update5 -> deps/test/lammps-22Jul2025
git clone --branch stable_22Jul2025_update5 --depth 1 \
  https://github.com/lammps/lammps.git deps/test/lammps-22Jul2025
```

The `deps/` directory holds large, machine-specific source and build trees and
is intentionally ignored by Git.

## 2. Create the MOFA Conda environment

```bash
module reset
module use /soft/modulefiles
module load conda
conda activate base
conda env create \
  --file envs/environment-polaris.yml \
  --prefix "$PWD/mofa_env"
conda activate "$PWD/mofa_env"
```

This installs MOFA, Redis, MongoDB, RASPA2, ChargeMol, MACE, and
`cupy-cuda12x==13.6.0` (the Kokkos ML-IAP device bridge requires CuPy 13.x, which
still supports `numpy<2`).

Verify the key versions and commands:

```bash
python --version                                  # 3.10.x
python -c "import redis; print('redis-py', redis.__version__)"   # 5.x
python -c "import cupy; print('CuPy', cupy.__version__)"          # 13.6.0
command -v redis-server mongod chargemol simulate monitor_utilization
```

CUDA device discovery is checked later from inside a PBS job — login nodes do not
expose a compute GPU.

## 3. Build CP2K

CP2K must be built on a compute node (its CUDA build steps need a GPU). Start an
interactive job, then run the build script, which requires `$PBS_JOBID`:

```bash
qsub -I -l select=1:system=polaris -l walltime=02:00:00 \
     -l filesystems=home:eagle -q debug -A ChemGraph
# inside the interactive job, from the repo root:
bash build-cp2k.sh
```

`build-cp2k.sh` configures the working Polaris module stack for CP2K:

- `PrgEnv-gnu` + `gcc-native/12.3` through the Cray compiler wrappers,
- `cuda/11.8` loaded **only** to satisfy `craype-accel-nvidia80`'s CPE-CUDA
  prerequisite (it provides the A100 target and CUDA-aware Cray MPI/GTL linkage),
  then unloaded,
- `cudatoolkit-standalone/12.8.1` for the actual toolchain and runtime,
- `cray-libsci` and `cray-fftw`.

It builds the CP2K toolchain, then the `local` and `local_cuda` `ssmp`/`psmp`
targets, and verifies each executable's CUDA 12.8.1 runpath, NVRTC, MPI GTL, and
LibSci linkage. Previous builds are moved aside with a `pre-cuda128-<timestamp>`
suffix rather than deleted.

The MOFA workflow uses this CUDA shell executable:

```text
deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp   # symlink -> cp2k.psmp
```

> **Why not `cuda/12.9`?** An earlier build used `cuda/12.9`, and CP2K then
> crashed at the first SCF step at runtime (surfacing in MOFA as
> `run_optimization ... AssertionError()`). The `cuda/11.8` prerequisite +
> `cudatoolkit-standalone/12.8.1` stack above is the one that runs correctly.
> The runtime module stack in `configs/polaris/polaris-repo.py`
> (`cp2k_worker_init`) must match the one used here.

Do not run a CUDA CP2K executable on the login node — CP2K initializes CUDA even
for `--version`, and the login node has no compute GPU (`cuInit` error 100).

## 4. Build LAMMPS

LAMMPS ML-IAP embeds Python and PyTorch, so it is built against its own venv, not
`mofa_env`. From an interactive compute-node job, at the repository root:

```bash
bash polaris-build/build-lammps-polaris.sh
```

`build-lammps-polaris.sh`:

- loads `PrgEnv-gnu`, `cuda`/`craype-accel-nvidia90`,
  `cudatoolkit-standalone/12.9.1`, `spack-pe-base cmake`, `cray-fftw`, and
  `conda` (base),
- creates `deps/test/lammps-22Jul2025/venv/` (Python 3.12 from
  `/soft/applications/conda`), installs `../python/wheel_requirements.txt`,
- builds with Kokkos+CUDA, `ML-IAP`, `ML-SNAP`, `PYTHON`, `FFT_KOKKOS=CUFFT`,
  `FFT_SINGLE=yes`, MPI on, and `make install-python`.

The resulting binary is:

```text
deps/test/lammps-22Jul2025/build-mliap-no-mpi/lmp
```

Then install the MACE stack into the **LAMMPS venv** (needed to build the model
in step 5 and for the ML-IAP runtime):

```bash
source deps/test/lammps-22Jul2025/venv/bin/activate
pip install mace-torch==0.3.13 cuequivariance-torch cuequivariance-ops-torch-cu12
python -c "import lammps; print(lammps.__file__)"   # resolves inside the venv
deactivate
```

## 5. Prepare the MACE model

Download the MACE-MP-0 medium model and convert it to the LAMMPS ML-IAP format,
using the LAMMPS venv (it has `cuequivariance` and `mace-torch`):

```bash
source deps/test/lammps-22Jul2025/venv/bin/activate
cd input-files/mace
bash get-macemp-0a.sh                 # downloads mace-mp0_medium
python create_lammps_model.py mace-mp0_medium --dtype float32 --format mliap
cd ../..
deactivate
```

This produces:

```text
input-files/mace/mace-mp0_medium-mliap_lammps.pt
```

`create_mliap_model.py` is a no-CuEquivariance fallback kept for environments
that pin Torch 2.1; the `--format mliap` path above is the one MOFA uses.

## 6. Executor environments

All executor environments are defined in `configs/polaris/polaris-repo.py` and
applied in `make_parsl_config()`.

- **CP2K executor** — uses `cp2k_worker_init`, which loads the same CP2K module
  stack as the build (`cuda/11.8` → `craype-accel-nvidia80` →
  `cudatoolkit-standalone/12.8.1`, `cray-libsci`, `cray-fftw`, `CUDA_PATH`,
  `MPICH_GPU_SUPPORT_ENABLED`) and activates `mofa_env`. The `dft_cmd` computed
  field launches `deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp` under
  `bin/set-affinity-gpu-polaris.sh` with `mpiexec -n 4 --ppn 4` on a single node
  (`nodes_per_cp2k=1`).
- **LAMMPS executor** — the Parsl worker uses the generic `worker_init`
  (`mofa_env`), but every `lmp` invocation goes through the wrapper
  `bin/run-lammps-polaris.sh`, which activates the **LAMMPS venv** and loads
  `cudatoolkit-standalone/12.9.1` in the child process. LAMMPS therefore runs
  against a different Python than the rest of MOFA on purpose: its ML-IAP C++ ABI
  and Torch must match that venv. Launch flags (`-k on g 1 -sf kk -pk kokkos ...`)
  come from `lammps_cmd`.

To change paths, module versions, or launch layout, edit
`configs/polaris/polaris-repo.py` (and `bin/run-lammps-polaris.sh` for the LAMMPS
runtime) — not the run scripts.

## 7. Validate CP2K standalone (optional but recommended)

Before the full workflow, confirm CP2K itself runs. From an interactive job:

```bash
# single node
bash cp2k-test/run-polaris.sh
# two nodes (needs -l select=2)
bash cp2k-test/run-polaris-2node.sh
```

Each writes a `run-*/cp2k.out`; success is the `PROGRAM ENDED AT` marker.

## 8. Run the workflow

One-node smoke test first, then the multi-node scaling test:

```bash
qsub run-polaris-local-smoke.sh
qstat -u "$USER"
# after it succeeds:
qsub run-polaris-repo-test.sh
```

Outputs land under `run/parallel-local-smoke-*` and `run/parallel-polaris-repo-*`.
Each run directory contains `run.log`, `params.json`, `compute-config.json`, the
MongoDB `db/`, service logs, per-node task logs, and results JSON. Confirm
`run.log` shows CP2K tasks succeeding (no `Task run_optimization failed ...
AssertionError()`) and that `dft-runs/mof-*-optimize-default/` hold non-empty
outputs.

## Troubleshooting

- **CP2K `AssertionError` / SCF crash in MOFA:** The CP2K runtime CUDA stack does
  not match the build. Ensure `cp2k_worker_init` in
  `configs/polaris/polaris-repo.py` loads `cuda/11.8` +
  `cudatoolkit-standalone/12.8.1` (not `cuda/12.9`), matching `build-cp2k.sh`.
- **CP2K `cuInit` error 100:** You ran a CUDA executable on the login node. Run
  inside a PBS compute job.
- **LAMMPS import/ABI errors:** LAMMPS must run from its own venv via
  `bin/run-lammps-polaris.sh`, not `mofa_env`. Rebuild with
  `polaris-build/build-lammps-polaris.sh` if the tree is stale.
- **ML-IAP `compute_forces failure` / `cupy` undefined:** Update `mofa_env` from
  `envs/environment-polaris.yml`; the bridge needs `cupy-cuda12x==13.6.0`.
- **The MACE model is missing:** Re-run step 5 inside the LAMMPS venv.

## Note on the canonical build scripts

The canonical, working builds are **`build-cp2k.sh`** (repo root) and
**`polaris-build/build-lammps-polaris.sh`**. An older
`polaris-build/build-cp2k-polaris.sh` remains in the tree but targets a different
CUDA stack (`cuda/12.9`) that did not run correctly here — prefer `build-cp2k.sh`.
