# Polaris CP2K diagnostic results — 2026-09-18

All four stages passed in job `7632220.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`.
PBS reported final state `F` and `Exit_status = 0`. The diagnostic ran from
13:51:05 to 14:01:19 UTC; PBS walltime, including setup, was 11 minutes 20 seconds.

Nodes: `x3001c0s19b1n0` and `x3001c0s1b1n0`, each with four A100-SXM4-40GB GPUs
and NVIDIA driver 580.65.06. Calculations used eight MPI ranks, four per node,
eight OpenMP threads per rank, and the workflow's CP2K command and GPU affinity.

| Stage | Result | Elapsed seconds | SCF steps |
| --- | --- | ---: | --- |
| Executable, MPS off | Passed | 137.273 | 44 |
| ASE shell, MPS off | Passed | 170.891 | 44, then 6 |
| Executable, MPS on | Passed | 131.273 | 44 |
| ASE shell, MPS on | Passed | 163.287 | 44, then 6 |

Both ASE stages used shell version 7.0 and the existing inline `SET_POS`
protocol with `set_pos_file=False`. Each evaluated energy, forces, and stress
for the 112-atom input, displaced one atom by 0.001 Angstrom, and repeated the
evaluation through the same shell process. All returned values were finite.
The installed ASE warning about MPI structures over 100 atoms appeared, but
neither stage stalled.

Direct energies were `-682.9169987513382` Hartree with MPS off and
`-682.9169987513379` Hartree with MPS on, a difference of approximately
`2.3e-13` Hartree. ASE returned `-18583.11659519` eV initially and
`-18583.11622071` eV after displacement, identical at the recorded precision
with MPS off and on. These timings are diagnostic observations, not a benchmark.

Six GPU health snapshots per node passed: initial, after each calculation,
and before MPS startup. They showed no reset/recovery requests or volatile
uncorrectable ECC errors. MPS logs confirm four connected CP2K client processes
and associations with all four GPU UUIDs on each node during each MPS stage.
Both private services accepted shutdown and logged `Exiting`; node manifests
record `mps_started: false`, and the cleanup launch returned zero.

The logs also contain MPS credential messages during connection setup and
receive-status 806 messages when clients exit. Successful client connections,
calculation exit codes, and shutdown were checked rather than treating those
strings alone as calculation failures.

## Diagnostic correction

The first job, `7631545`, completed the executable and both ASE evaluations
with MPS off, but its validator rejected ASE output because the DBCSR startup
device-count banner was absent. This stopped that job before the MPS stages.

The validator now accepts positive call counts and execution times for GPU FFT
routines when shell mode omits that banner. Direct runs still require the
device-count banner; explicit invalid device counts are rejected. A regression
test covers banner-free shell output, missing GPU execution evidence, zero
calls/timing, and an explicit zero device count. All five diagnostic tests passed;
the PBS script syntax and `git diff --check` also passed before resubmission.
The successful second job used this correction throughout all four stages.

## Evidence and scope

The successful run directory (`run/cp2k-diagnostic-7632220.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov-1789739412282509203/`, local evidence; not included in Git)
contains `summary.json`, per-stage scientific/protocol/stderr logs,
`ase-result.json`, node health snapshots, MPS logs, and `review.json` with
client counts, energy comparisons, and shutdown verification.

The first run directory (`run/cp2k-diagnostic-7631545.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov-1789711764881310337/`, local evidence; not included in Git)
retains its original failed validator result and all original logs.
`revalidation.json` separately records the corrected validation of both stages.
PBS logs are `cp2k-diagnostic.o7632220` and `cp2k-diagnostic.e7632220` in the
repository root.

This completes the requested two-node executable and ASE-shell diagnostic.
It supports using the existing CP2K build and launch configuration on these
healthy nodes. No CP2K rebuild, GPU reset, or production workflow change was
needed. It does not establish recovery of previously unhealthy nodes or validate
Parsl scheduling, complete geometry optimizations, simultaneous LAMMPS/CP2K
workloads, or every generated MOF.
