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

> **The site `conda` module is broken.** `module load conda` used to provide a
> working Conda install with a Python 3.12 base env, but its module collection
> has since had all its modules removed and no longer works. Every script in
> this guide instead sources a local Miniconda install directly:
> `/lus/eagle/projects/datascience/hari/.local/miniconda3` (this is also set up
> in `~/.bashrc`). If you are a different user, install your own Miniconda and
> update the paths below (and in `build-cp2k.sh`, `build-lammps.sh`,
> `bin/run-lammps-polaris.sh`, and `configs/polaris/polaris-repo.py`)
> accordingly.

## Prerequisites

You need:

- A Polaris account and a project allocation.
- A clone of this repository on a filesystem visible to the compute nodes
  (Eagle or Flare).
- Network access from the login node to fetch CP2K, LAMMPS, and the MACE model.
- A working local Miniconda install (see the note above). Create it once with:

  ```bash
  # only if you don't already have one
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -p "$HOME/.local/miniconda3"
  ```

The PBS run scripts set `#PBS -A ChemGraph`. Replace `ChemGraph` with your own
Polaris allocation in these files before submitting:

```text
run-polaris-local-smoke.sh
run-polaris-repo-test.sh
```

`build-cp2k.sh` runs inside an interactive PBS job (see below), so it has no
`#PBS -A` line to edit. The LAMMPS build (`polaris-build/build-lammps.sh`) also
runs interactively.

## 1. Download the CP2K and LAMMPS sources

Use the versions validated by this repository: **CP2K 2025.2** and **LAMMPS
`stable_22Jul2025` (update5)**.

```bash
mkdir -p deps deps/test

# CP2K 2025.2 -> deps/cp2k
git clone --branch polaris_build_sep_2026 --depth 1 \
  git@github.com:harikrishna1410/cp2k.git

# LAMMPS 22Jul2025 update5 -> deps/test/lammps-22Jul2025
git clone --branch stable_22Jul2025_update5 --depth 1 \
  https://github.com/lammps/lammps.git deps/test/lammps-22Jul2025
```

The `deps/` directory holds large, machine-specific source and build trees and
is intentionally ignored by Git.

## 2. Create the MOFA Conda environment

```bash
source /lus/eagle/projects/datascience/hari/.local/miniconda3/etc/profile.d/conda.sh
conda activate base
conda env create \
  --file envs/environment-polaris.yml \
  --prefix "$PWD/mofa_env"
conda activate "$PWD/mofa_env"
pip install --no-deps -e .
```

This installs Redis, MongoDB, RASPA2, ChargeMol, MACE, and
`cupy-cuda12x==13.6.0` (the Kokkos ML-IAP device bridge requires CuPy 13.x, which
still supports `numpy<2`). `envs/environment-polaris.yml` only pins the `mofa`
package's runtime dependencies (colmena, parsl, ase, pymatgen, etc.) — it does
not install `mofa` itself, so the final `pip install --no-deps -e .` is
required. Without it, `mofa`'s console-script entry points (e.g.
`monitor_utilization`, used by `launch_monitor_process` in
`configs/polaris/polaris-repo.py`) will not exist in `mofa_env/bin`, and
`run_parallel_workflow.py` will fail with `FileNotFoundError: [Errno 2] No such
file or directory: 'monitor_utilization'`. `--no-deps` is required here: without
it, pip would try to resolve `pyproject.toml`'s `openbabel-wheel` dependency
(not `openbabel`, the conda-forge package this env actually installs) and could
shadow it with a mismatched build.

Verify the key versions and commands:

```bash
python --version                                  # 3.10.x
python -c "import redis; print('redis-py', redis.__version__)"   # 5.x
python -c "import cupy; print('CuPy', cupy.__version__)"          # 13.6.0
command -v redis-server mongod chargemol simulate monitor_utilization
```

CUDA device discovery is checked later from inside a PBS job — login nodes do not
expose a compute GPU. `mofa_env`'s compiled dependencies (RASPA2, ChargeMol,
Torch, CuPy) resolve their own bundled libraries and do not need any module
loaded at all — see the [Executor environments](#executor-environments) note on
`worker_init` below.

## 3. Build CP2K

CP2K must be built on a compute node (its CUDA build steps need a GPU). Start an
interactive job, then run the build script, which requires `$PBS_JOBID`:

```bash
qsub -I -l select=1:system=polaris -l walltime=02:00:00 \
     -l filesystems=home:eagle -q debug -A ChemGraph
# inside the interactive job, from the repo root:
bash polaris-build/build-cp2k.sh
```

`build-cp2k.sh` configures the working Polaris module stack for CP2K:

- `PrgEnv-gnu` + `gcc-native/14` through the Cray compiler wrappers,
- `cuda` loaded **only** to satisfy `craype-accel-nvidia80`'s CPE-CUDA
  prerequisite (it provides the A100 target and CUDA-aware Cray MPI/GTL linkage),
  then unloaded,
- `cudatoolkit-standalone/13.0.1` for the actual toolchain and runtime,
- `cray-libsci` and `cray-fftw`.

It builds the CP2K toolchain, then the `local` and `local_cuda` `ssmp`/`psmp`
targets, and verifies each executable's CUDA 13.0.1 runpath, NVRTC, MPI GTL, and
LibSci linkage. Previous builds are moved aside with a timestamped suffix rather
than deleted.

The MOFA workflow uses this CUDA shell executable:

```text
deps/cp2k/exe/local_cuda/cp2k_shell.psmp   # symlink -> cp2k.psmp
```

> **Why CUDA 13.0.1, not `cuda/12.9`?** Earlier attempts used various CUDA 12.x
> stacks, but `cray-mpich/9.1.0`'s `libmpi_gtl_cuda.so` has `NEEDED
> libcudart.so.13`, so linking against any CUDA 12.x runtime leaves the whole
> toolchain broken starting at the OpenBLAS `getarch` probe (since
> `craype-accel-nvidia80` injects `-lmpi_gtl_cuda` into every link). CUDA 13.0.1
> also fixes `nvcc`'s inability to parse `gcc-native/14`'s headers (`_Float128`,
> `0.0bf16`). `build-cp2k.sh` runs a preflight link + `nvcc` compile before
> spending ~25 minutes on dependencies, specifically to catch this class of
> failure early if ALCF changes a module default again.
>
> The runtime wrapper `bin/run-cp2k-polaris.sh` (see [Executor
> environments](#executor-environments)) must load the same module stack used
> here.

Do not run a CUDA CP2K executable on the login node — CP2K initializes CUDA even
for `--version`, and the login node has no compute GPU (`cuInit` error 100).

> Two older CP2K build scripts remain in `polaris-build/` for reference —
> `build-cp2k-polaris.sh` and `build-cp2k_v2.sh` — both target CP2K 2025.1 with
> `gcc-native/12.3`, which is no longer installed on the system. Use
> `build-cp2k.sh`.

## 4. Build LAMMPS

LAMMPS ML-IAP embeds Python and PyTorch, so it is built against its own venv, not
`mofa_env`. This is a hard constraint, not a preference: `mofa_env` pins
`torch==2.1.0` (so the rest of MOFA is unaffected by CuEquivariance's
`torch>=2.4` requirement, and MACE falls back to its e3nn implementation there),
while LAMMPS's ML-IAP needs `torch==2.5.0` with CuEquivariance for accelerated
MACE inference. LAMMPS's `PKG_PYTHON`/`MLIAP_ENABLE_PYTHON` build links directly
against a specific Python + libtorch at build time (a C++ ABI dependency, not
just an import), so the two Torch versions cannot share one environment.

The LAMMPS venv also needs Python 3.12, which the (now-broken) site `conda`
module used to provide. The local Miniconda's own `base` environment is Python
3.14, which has no `torch==2.5.0` wheels, so create a dedicated Python 3.12 env
once:

```bash
source /lus/eagle/projects/datascience/hari/.local/miniconda3/etc/profile.d/conda.sh
conda create -n lammps-py312 python=3.12
```

Then, from an interactive compute-node job, at the repository root:

```bash
bash polaris-build/build-lammps.sh
```

`build-lammps.sh` mirrors `build-cp2k.sh`'s module stack — LAMMPS is built with
`BUILD_MPI=ON` through the Cray compiler wrappers (`cc`/`CC` via
`nvcc_wrapper`), so it hits the same `libmpi_gtl_cuda.so` → `libcudart.so.13`
constraint CP2K does:

- `PrgEnv-gnu` + `gcc-native/14`, `cray-libsci`, `cray-fftw`,
- `cuda` → `craype-accel-nvidia80` → `cudatoolkit-standalone/13.0.1` (Polaris's
  GPUs are A100s, so this must be `nvidia80`, not `nvidia90`),
- `spack-pe-base cmake`,
- the local Miniconda's `lammps-py312` env (only used to seed the venv's
  Python; conda deactivates once the venv is created).

It then:
- creates `deps/test/lammps-22Jul2025/venv/` (Python 3.12), installs
  `../python/wheel_requirements.txt`,
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

> An older build script, `polaris-build/build-lammps-polaris.sh`, remains in
> the tree for reference but uses `craype-accel-nvidia90` (the H100 target;
> Polaris has A100s) and `module load conda` (broken). Use `build-lammps.sh`.

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

- **Shared `worker_init`** — every Parsl executor (CP2K, LAMMPS, inference,
  training, RASPA2, helper) uses the same `worker_init`: `module reset`,
  `module load cray-mpich-abi`, then activating `mofa_env` from the local
  Miniconda. It does **not** load a `gcc` or `cudatoolkit-standalone` module —
  `mofa_env`'s compiled dependencies (RASPA2, ChargeMol, Torch, CuPy) resolve
  their own bundled libraries with no module loaded at all, and `module load
  gcc` actively breaks things: it deactivates `cray-mpich`/`cray-libsci`
  (Lmod prints an "Inactive Modules" warning when this happens). The one
  module `worker_init` does need is `cray-mpich-abi`, because `mofa_env`'s
  `mpi4py` (used by Colmena/Parsl) expects the standard-ABI `libmpi.so.12`,
  which the default `cray-mpich` module does not provide on its own (it only
  ships `libmpi_nvidia.so.12`).
- **CP2K executor** — `worker_init` only activates `mofa_env`; the CP2K-specific
  module stack (same as the build: `PrgEnv-gnu`, `gcc-native/14`, `cray-libsci`,
  `cray-fftw`, `cuda` → `craype-accel-nvidia80` → `cudatoolkit-standalone/13.0.1`,
  `CUDA_PATH`, `MPICH_GPU_SUPPORT_ENABLED`) is loaded fresh by the wrapper
  `bin/run-cp2k-polaris.sh`, one per MPI rank, mirroring how LAMMPS is handled
  below. The `dft_cmd` computed field launches
  `bin/run-cp2k-polaris.sh` (which execs
  `deps/cp2k/exe/local_cuda/cp2k_shell.psmp`) under
  `bin/set-affinity-gpu-polaris.sh`, spanning `nodes_per_cp2k` nodes (2 by
  default) at `--ppn {gpus_per_node}` ranks per node.
- **LAMMPS executor** — the Parsl worker uses the same shared `worker_init`
  (`mofa_env`), but every `lmp` invocation goes through the wrapper
  `bin/run-lammps-polaris.sh`, which loads the same module stack
  `build-lammps.sh` uses (`PrgEnv-gnu`, `gcc-native/14`, `cray-libsci`,
  `cray-fftw`, `craype-accel-nvidia80`, `cudatoolkit-standalone/13.0.1`) and
  then activates the **LAMMPS venv**. LAMMPS therefore runs against a different
  Python than the rest of MOFA on purpose: its ML-IAP C++ ABI and Torch must
  match that venv. Launch flags (`-k on g 1 -sf kk -pk kokkos ...`) come from
  `lammps_cmd`.

To change paths, module versions, or launch layout, edit
`configs/polaris/polaris-repo.py` (and `bin/run-cp2k-polaris.sh` /
`bin/run-lammps-polaris.sh` for the respective runtime module stacks) — not the
run scripts.

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

> `run-polaris-local-smoke.sh` and `run-polaris-repo-test.sh` still use
> `module load conda` to bootstrap their own shell before Parsl's
> `worker_init` takes over per-executor. If the site `conda` module is broken
> in your environment, update these to source the local Miniconda directly
> too, the same way `worker_init` does.

## Troubleshooting

- **CP2K `AssertionError` / SCF crash in MOFA:** The CP2K runtime CUDA stack
  does not match the build. Ensure `bin/run-cp2k-polaris.sh` loads
  `cudatoolkit-standalone/13.0.1` via `craype-accel-nvidia80`, matching
  `build-cp2k.sh`.
- **CP2K `cuInit` error 100:** You ran a CUDA executable on the login node. Run
  inside a PBS compute job.
- **LAMMPS import/ABI errors:** LAMMPS must run from its own venv via
  `bin/run-lammps-polaris.sh`, not `mofa_env`. Rebuild with
  `polaris-build/build-lammps.sh` if the tree is stale.
- **ML-IAP `compute_forces failure` / `cupy` undefined:** Update `mofa_env` from
  `envs/environment-polaris.yml`; the bridge needs `cupy-cuda12x==13.6.0`. Note
  that `build-lammps.sh`'s LAMMPS venv installs a plain, unpinned
  `cupy-cuda12x` alongside `cudatoolkit-standalone/13.0.1` — this has not been
  reconciled to `cupy-cuda13x` yet; treat that venv's CuPy version as unverified
  if you hit device-bridge issues there specifically.
- **The MACE model is missing:** Re-run step 5 inside the LAMMPS venv.
- **`ImportError: libmpi.so.12: cannot open shared object file`** (or any
  `mpi4py` import failure) **in a Parsl worker:** The default `cray-mpich`
  module does not provide the standard-ABI `libmpi.so.12` that `mofa_env`'s
  `mpi4py` expects. Ensure `worker_init` loads `cray-mpich-abi`, not plain
  `cray-mpich`.
- **`module load conda` fails / "unknown module"**: The site `conda` module's
  collection has been broken (all its modules were removed). Source your local
  Miniconda's `etc/profile.d/conda.sh` directly instead, as done throughout
  this guide.

## Note on the canonical build scripts

The canonical, working builds are **`polaris-build/build-cp2k.sh`** and
**`polaris-build/build-lammps.sh`**. Older scripts remain in the tree for
reference but should not be used:

- `polaris-build/build-cp2k-polaris.sh` and `polaris-build/build-cp2k_v2.sh`
  target CP2K 2025.1 with `gcc-native/12.3`, which no longer exists on the
  system.
- `polaris-build/build-lammps-polaris.sh` targets `craype-accel-nvidia90` (the
  H100 target; Polaris has A100s) and uses the broken `module load conda`.
