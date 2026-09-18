# Standalone Polaris LAMMPS/MPS diagnostic

This test uses the external LAMMPS build and the same runtime wrapper as MOFA,
without Parsl, Redis, MongoDB, generation, or CP2K. The default source is
`/lus/eagle/projects/ChemGraph/thang/soft/lammps/test`. Original inputs and previous
results are never overwritten. No GPU resets or package/build changes are made.

## Submit from the repository root

```bash
qsub lammps-test/run-polaris.sh
```

The job requests one Polaris node for one hour in `debug`, account `ChemGraph`.
Results are written immediately to `run/lammps-diagnostic-<job>-<timestamp>/`.
PBS stdout prints this path and progress. Use a fresh allocation: an existing MPS
daemon causes the diagnostic to stop, rather than change another application's
MPS service. The launch commands and GPU UUIDs are recorded in the results.

The six stages are:

| Stage | Input | MPS | Clients/GPU | Total clients |
| --- | --- | --- | ---: | ---: |
| 1 | Lennard-Jones, CUDA/Kokkos baseline | Off | 1 | 4 |
| 2 | Your MACE input | Off | 1 | 4 |
| 3 | Your MACE input | On | 1 | 4 |
| 4 | Your MACE input | On | 2 | 8 |
| 5 | Your MACE input | On | 3 | 12 |
| 6 | Your MACE input | On | 4 | 16 |

Short MACE copies use five minimization iterations, 20 MD steps, and output every
step. Processes run directly, without MPI, using `-k on g 1 -sf kk -pk kokkos
newton on neigh half`. Each has its own directory, GPU UUID, and distinct allowed
CPU, with one OpenMP/BLAS thread. These are correctness/concurrency diagnostics,
not a tuned CPU-affinity benchmark of the full workflow.

GPU memory, utilization and temperature are sampled every five seconds. The MPS
client list must show the requested processes connected simultaneously; merely
having a daemon or successful process exit is insufficient. A failed GPU is
excluded from later stages, and a partial-node result cannot count as a full-node
pass. An explicit GPU recovery/reset request stops the test. Each short stage has
a ten-minute timeout, with a 55-minute overall execution deadline followed by
cleanup. Timeouts do not establish a deadlock.

## Read the results

- `summary.json`: highest verified concurrency across all four GPUs, stage
  results, exit codes, elapsed times, observed overlap,
  MD loop times, and aggregate MD steps per wall-clock second. The aggregate rate
  includes model loading and minimization; compare the same MACE input across
  stages, not against the LJ baseline.
- `health/`: original `nvidia-smi` inventory and full XML before/after stages.
- `<stage>/gpu-<index>-client-<n>/`: isolated input/data, stdout, stderr,
  `log.lammps`, trajectory/restart outputs and `result.json`.
- `<stage>/gpu-samples.jsonl`: GPU samples and connected MPS client PIDs.
- `mps-control.jsonl`, `mps-start.json`, and `mps-logs/`: startup, connectivity,
  daemon/server logs, and shutdown evidence. Logs are also copied after each stage.

Failures are classified as out of memory, MPS error, CUDA error, simulation
error, process error, invalid output, timeout, or unobserved MPS overlap.
Acceptance requires finite thermodynamic output, all requested MD steps, normal
LAMMPS termination, actual MPS overlap, and unchanged healthy GPU inventory.
The job returns nonzero for any failure, inconclusive overlap, or incomplete test.

If LJ fails, investigate GPU/driver/CUDA/Kokkos. If only MACE fails without MPS,
investigate ML-IAP, its Python environment, the model or input. If only MPS stages
fail, inspect the MPS logs and device mapping. Higher-concurrency failures can
also mean insufficient GPU memory. Passing on a different node does not establish
that `x3003c0s25b0n0` or `x3208c0s37b1n0` has recovered, or validate CP2K/MPI.

## What this does and does not establish for MOFA and CP2K

The executable, runtime wrapper, and MACE model match MOFA, but several test
conditions differ: a fresh node, private MPS directories, UUID-based device
assignment, direct subprocess launch instead of Parsl/MPI placement, and a fixed
structure instead of generated MOFs. The current source has 976 atoms; MOFA
replicates each generated structure into a 2x2x2 supercell and can therefore have
different memory requirements. The short diagnostic also runs only 20 MD steps,
compared with the current workflow's 1,000. A standalone pass cannot identify
which difference explains a workflow failure, nor establish that every MOF fits
at that concurrency. Compare failures by hostname and error type.

This harness does not execute CP2K. It checks the GPU/MPS layer that CP2K also
uses, but not distributed MPI initialization or ASE's `cp2k_shell` protocol.
`cp2k-test/run-polaris-2node.sh` provides an existing eight-rank, two-node CP2K
executable baseline inside a separate allocation. A subsequent ASE shell test
through the workflow's launch configuration is needed to cover the original
`AssertionError` path. A CUDA initialization failure followed by an ASE assertion
should be diagnosed from the child process's stderr, not the assertion alone.

## Confirm with the original input length

After inspecting the short-test results, submit a separate confirmation at the
highest passing concurrency (replace `4` if necessary):

```bash
qsub -v LAMMPS_TEST_FULL=1,LAMMPS_TEST_MAX_PER_GPU=4 lammps-test/run-polaris.sh
```

Full mode runs the LJ baseline, the original MACE minimization/MD lengths without
MPS, then those same lengths at only the selected MPS concurrency. The current
source input has 100 minimization iterations and 100 MD steps. Thermodynamic
logging stays at every step. Full stages allow 30 minutes each, still subject to
the overall 55-minute execution deadline. Incomplete runs are not passes.

`LAMMPS_ROOT` and `LAMMPS_VENV` select the external installation. Set
`LAMMPS_TEST_DIR` to another directory containing compatible `in.lammps` and
`data.lmp` files. The model must already exist; no downloads are performed.

Inside an existing **fresh one-node** allocation, from the repository root:

```bash
bash lammps-test/run-polaris.sh --max-per-gpu 4
```

For a local input check without any GPU commands, modules, or PBS submission:

```bash
/lus/eagle/projects/ChemGraph/thang/soft/lammps/.venv/bin/python \
  lammps-test/diagnose.py --prepare-only
/lus/eagle/projects/ChemGraph/thang/soft/lammps/.venv/bin/python \
  -m unittest discover -s tests -p test_lammps_diagnostic.py
```

The controller uses only Python's standard library (Python 3.9 or newer); the
login node's system `python3` can be too old. `--output` selects a new
result directory; existing directories are never reused. The direct Python CLI
also supports `--input-dir`, `--wrapper`, `--full`, and `--stage-timeout`.
