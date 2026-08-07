# Installing and Running MOFA on Polaris

This guide creates a repository-local MOFA installation on Polaris. CP2K and
LAMMPS are built under `deps/`, and both MOFA and the LAMMPS Python interface
use the same Python 3.10 Conda environment at `./mofa_env`.

Run all commands from the MOFA repository root unless a step says otherwise.

## Prerequisites

You need:

- A Polaris account and a project allocation.
- A clone of this repository on a filesystem available to compute nodes.
- Network access from the login node to clone CP2K and LAMMPS and download the
  MACE model.

The supplied PBS scripts contain the placeholder `YOUR_PROJECT`. Replace it
with your Polaris allocation in these four files before submitting jobs:

```text
polaris-build/build-cp2k-polaris.sh
polaris-build/build-lammps-polaris.sh
run-polaris-local-smoke.sh
run-polaris-repo-test.sh
```

You can locate every placeholder with:

```bash
grep -n 'YOUR_PROJECT' \
  polaris-build/build-cp2k-polaris.sh \
  polaris-build/build-lammps-polaris.sh \
  run-polaris-local-smoke.sh \
  run-polaris-repo-test.sh
```

## 1. Download CP2K and LAMMPS

Use the versions validated by this repository: CP2K 2025.1 and LAMMPS
`stable_29Aug2024_update3`.

```bash
mkdir -p deps
git clone --branch v2025.1 --depth 1 \
  https://github.com/cp2k/cp2k.git deps/cp2k-2025.1
git clone --branch stable_29Aug2024_update3 --depth 1 \
  https://github.com/lammps/lammps.git deps/lammps
```

The `deps/` directory contains large, machine-specific source and build trees
and is intentionally ignored by Git.

## 2. Build CP2K

Submit the CP2K build from the repository root:

```bash
qsub polaris-build/build-cp2k-polaris.sh
```

Monitor it with:

```bash
qstat -u "$USER"
```

The job configures the Polaris GNU/Cray compiler stack, builds CPU and CUDA
`ssmp` and `psmp` variants, and verifies the executables. The MOFA workflows
use these two CUDA shell executables:

```text
deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.ssmp
deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp
```

Do not use a CUDA CP2K executable as a login-node verification command. CP2K
initializes CUDA even for `--version`, and the login node does not expose a
compute GPU. The PBS build performs this check on its allocated compute node.

## 3. Create the MOFA Conda Environment

Load Conda and create the environment at the path expected by the build and
run scripts:

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

This installs MOFA, Redis, MongoDB, RASPA2, ChargeMol, MACE, and the Python
build dependencies needed by LAMMPS. It intentionally installs the MOFA
runtime package without the test dependency extra.

Check the key versions and commands:

```bash
python --version
python -c "import redis; print('redis-py', redis.__version__)"
command -v redis-server mongod chargemol simulate monitor_utilization
```

Python should be version 3.10, and redis-py should be version 5.x.

## 4. Build LAMMPS in the MOFA Environment

Submit the LAMMPS build only after `./mofa_env` exists:

```bash
qsub polaris-build/build-lammps-polaris.sh
qstat -u "$USER"
```

The build enables CUDA, Kokkos, ML-IAP, ML-SNAP, and Python. It installs the
LAMMPS Python package into `./mofa_env` and links the LAMMPS executable against
that environment's `libpython3.10`. Do not create or activate a separate
LAMMPS virtual environment.

After the job succeeds, load its runtime modules and verify the installation:

```bash
module reset
module use /soft/modulefiles
module load gcc
module load cudatoolkit-standalone/12.8
module load conda
conda activate base
conda activate "$PWD/mofa_env"

python -c "import lammps; print(lammps.__file__)"
ldd deps/lammps/build-mliap/lmp | grep "$PWD/mofa_env/lib/libpython3.10"
for package in ML-IAP ML-SNAP KOKKOS PYTHON; do
  deps/lammps/build-mliap/lmp -help | grep -qw "$package" && \
    echo "$package: enabled"
done
```

The Python import and linked Python library must both resolve inside
`$PWD/mofa_env`.

## 5. Prepare the MACE Model

With `mofa_env` active, download the MACE-MP model and convert it for LAMMPS
ML-IAP:

```bash
(
  cd input-files/mace
  bash get-macemp-0a.sh
)
```

This creates:

```text
input-files/mace/mace-mp0_medium-mliap_lammps.pt
```

The conversion uses MACE's e3nn implementation because the Polaris
environment retains Torch 2.1 and does not install CuEquivariance 0.4.

## 6. Run the One-Node Smoke Test

First run the inexpensive end-to-end smoke job:

```bash
qsub run-polaris-local-smoke.sh
qstat -u "$USER"
```

The job checks CP2K, LAMMPS linkage, the MACE model, Redis, MongoDB, ChargeMol,
RASPA2, and MOFA before starting the workflow. It uses small generation and
simulation settings on one Polaris node.

After completion, inspect the PBS output and the newest run directory:

```bash
ls -dt run/parallel-local-smoke-* | head -n 1
```

The run directory includes `run.log`, `params.json`, `compute-config.json`,
the MongoDB `db/`, service logs, task logs, and simulation results.

## 7. Run the Eight-Node Repository Test

Submit the scaling test only after the one-node smoke test succeeds:

```bash
qsub run-polaris-repo-test.sh
qstat -u "$USER"
```

This job uses eight nodes, enables NVIDIA MPS, launches distributed CP2K,
runs GPU LAMMPS/ML-IAP workers, and writes LAMMPS scratch data to node-local
RAM disks. Its output is stored under a directory matching:

```text
run/parallel-polaris-repo-*
```

## Troubleshooting

- **PBS rejects the job:** Confirm that every `YOUR_PROJECT` placeholder was
  replaced with an allocation available to your Polaris account.
- **A dependency source is missing:** Confirm the CP2K and LAMMPS clones use
  the exact paths from step 1.
- **`libcudart.so.12` is missing:** Load the CUDA 12.8 module before invoking
  the LAMMPS executable directly. The PBS scripts load it automatically.
- **LAMMPS links to another Python:** Remove or rename the stale
  `deps/lammps/build-mliap` build directory, then resubmit the LAMMPS build
  with `./mofa_env` present. Preserve the old directory if its contents are
  needed.
- **CP2K reports `cuInit` error 100:** Run the CUDA executable inside a PBS
  compute job, not on the login node.
- **The MACE model is missing:** Activate `mofa_env` and repeat step 5.
