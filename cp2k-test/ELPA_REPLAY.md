# Replaying the structure-dependent ELPA crash

Source workflow: `run/parallel-polaris-repo-18Sep26141032-b42970/`.
Replay completed as PBS job `7632347`, with exit status 0, on 2026-09-18.
The diagnostic ran 14:46:56–15:01:27 UTC on `x3005c0s19b0n0` and
`x3005c0s19b1n0`. PBS walltime including setup was 15 minutes 44 seconds.
All six baseline replays reproduced the same ELPA SIGFPE: both inputs in fresh
direct processes with MPS off and on, and both in fresh ASE shells with MPS on.
Thus shell reuse, MPS, and the original compute node are not required triggers.
Both CPU-ELPA controls and both GPU block-size-32 controls completed with converged
SCF (29 steps for `mof-521f5ced`, 35 for `mof-b0c05282`). Energies agree to numerical
precision between controls. The final two-step optimization of both MOFs in one
reused ASE shell with GPU block size 32 also completed; all six SCF evaluations
converged. Neither structure reached the geometry force tolerance in two steps.

| Test | mof-521f5ced | mof-b0c05282 |
| --- | --- | --- |
| Fresh direct process, original settings, MPS off | ELPA SIGFPE | ELPA SIGFPE |
| Fresh direct process, original settings, MPS on | ELPA SIGFPE | ELPA SIGFPE |
| Fresh ASE shell, original settings, MPS on | ELPA SIGFPE | ELPA SIGFPE |
| CPU ELPA, original block size 64, MPS on | SCF converged in 29 steps | SCF converged in 35 steps |
| GPU ELPA, block size 32, MPS on | SCF converged in 29 steps | SCF converged in 35 steps |
| Two LBFGS steps each, same ASE shell, GPU block size 32 | 3 converged SCFs: 29, 15, 15 steps | 3 converged SCFs: 35, 15, 18 steps |

Direct CPU/GPU-control energies are `-655.0917531729258` Hartree for
`mof-521f5ced`, and approximately `-706.705208326889` Hartree for `mof-b0c05282`.
CPU/GPU differences are zero and `1.14e-13` Hartree, respectively.
The final two-step ASE energies are `-17826.348370759` and `-19230.620350277` eV.
Both cases used shell PID 1613111, shell version 7.0, with inline coordinate
transfer. Energies, forces and stress remained finite.

Thirteen GPU health snapshots on each node showed no reset/recovery requests or
volatile uncorrectable ECC errors. Both MPS services shut down successfully;
their final logs contain `Exiting`, stop commands returned zero, and node manifests
record `mps_started: false`. The six intentional crash reproductions account for
the failed process exits inside this otherwise completed diagnostic.

The original sequence was `mof-42c8dfa7` (returned), `mof-521f5ced` (crashed),
new shell, `mof-573fd5f8` (returned), `mof-b0c05282` (crashed).
Both crashes occurred before the first SCF iteration completed. Neither failed
MOF produced a first optimization trajectory frame.

## Core dump findings

Both saved core dumps stop at the same instruction:
`launch_my_unpack_c_cuda_kernel_real_double+93: idiv %ebp`, with `ebp=0`.
ELPA's Fortran frames have `l_nev=0`, `stripe_width=0`, `stripe_count=0`,
`nblk=64`, four process columns, and `my_pcol=3`. These are CPU integer
division failures inside ELPA's CUDA launch wrapper.

| MOF | Atoms | Basis functions | Requested eigenvectors | Vectors in process columns 0–3 | Original outcome |
| --- | ---: | ---: | ---: | --- | --- |
| mof-42c8dfa7 | 120 | 1248 | 197 | 64, 64, 64, 5 | Returned |
| mof-521f5ced | 102 | 1126 | 181 | 64, 64, 53, 0 | SIGFPE |
| mof-573fd5f8 | 117 | 1249 | 199 | 64, 64, 64, 7 | Returned |
| mof-b0c05282 | 104 | 1176 | 192 | 64, 64, 64, 0 | SIGFPE |
| Previous standalone diagnostic | 112 | 1224 | 194 | 64, 64, 64, 2 | Passed |

The column counts follow ELPA's block-cyclic distribution with block size 64;
the empty last column is independently confirmed in both cores. The installed
ELPA 2024.05.001 source sets `stripe_width=0` when `l_nev=0` at
`src/elpa2/elpa2_trans_ev_tridi_to_band_template.F90:379`.
`src/GPU/CUDA/cuUtils_template.cu:325` then sets `blocksize=0` and evaluates
`stripe_width / blocksize` at line 331. The CPU instruction and zero divisor
agree with that source path. The sources are under
`/lus/eagle/projects/ChemGraph/thang/soft/cp2k/tools/toolchain/build/elpa-2024.05.001/`.

The CP2K guard in `src/fm/cp_fm_diag_utils.F` checks column ownership of the
full matrix. Here every process column owns part of the full basis, while the
requested occupied eigenvectors leave the last process column empty.

Input snapshots, SHA-256 provenance, and GDB transcripts are in
[`elpa-cases-18Sep26141032/`](elpa-cases-18Sep26141032/).
Original cores and scientific logs remain in the source run's temporary folders.

## Compute experiment

Submit from the repository root with:

```bash
qsub cp2k-test/run-elpa-replay-polaris.sh
```

Job `7632347` is complete; no replay remains pending. The script uses the same
eight-rank, two-node launch and eight threads as the workflow. It runs:

1. Both failed inputs in fresh direct CP2K processes with MPS off.
2. Both failed inputs in fresh direct and ASE processes with MPS on.
3. Both inputs with `GLOBAL / ELPA_KERNEL GENERIC` (CPU ELPA only; other GPU code remains enabled).
4. Both inputs with `GLOBAL / FM / NROW_BLOCKS 32` and `NCOL_BLOCKS 32` (GPU ELPA retained).
5. If a control returns valid output, both failed MOFs with two LBFGS steps each
   in a single reused ASE shell using that setting, preferring GPU block size 32.

The replay preserves `IGNORE_CONVERGENCE_FAILURE`, the original 128-step SCF
limit, geometry, basis, potential, cutoff, and functional. `PRINT_ELPA ON` is
added to all variants to expose the actual solver layout. Source inputs remain
unchanged. ASE uses its regular inline position transfer; only its input-generation
method is overridden to return the saved CP2K input without duplicating coordinates.

Expected crashes are recorded and the diagnostic continues if node health remains
good. Timeouts stop further launches. SCF convergence is reported separately
from process completion; two LBFGS steps do not establish full geometry convergence.
The original `mof-573fd5f8` task had an unconverged first SCF despite returning
successfully to MOFA, which is a separate issue from these SIGFPE crashes.

Results go to `run/cp2k-elpa-replay-<job>-<timestamp>/`, with original input copies,
per-stage commands, child stderr, ASE protocol, SCF outcomes, GPU health, and private
MPS logs. New core dumps are disabled to avoid duplicating the existing large cores.
No production workflow or external CP2K/ELPA build is modified by this experiment.

## Tested settings and limits

The GPU-preserving setting tested for these two MOFs is:

```text
&GLOBAL
  &FM
    NROW_BLOCKS 32
    NCOL_BLOCKS 32
  &END FM
&END GLOBAL
```

The successful outputs explicitly report `NBLK=32` and `NVIDIA_GPU`.
With four process columns, the requested eigenvector distributions become
`64, 53, 32, 32` and `64, 64, 32, 32`; none are empty. This is a tested workaround
for these inputs, not a general fix for ELPA's zero-width case: still smaller
eigenvector counts or different MPI layouts can leave empty partitions again.

The independently tested alternative is `ELPA_KERNEL GENERIC` inside `GLOBAL`.
It uses CPU ELPA while retaining the existing GPU-enabled CP2K build and other
GPU operations. It converged for both inputs with their original block size 64.
Its two-step optimization was not tested; the shared-shell optimization used
GPU block size 32. These settings were applied only to diagnostic copies.

The source-level defect remains in the installed ELPA library. No external
library patch, rebuild, or production configuration change was made. The results
establish the cause of these SIGFPE crashes; they do not establish that two
optimization steps suffice to relax the structures for downstream science.

## Saved results

The completed run directory (`run/cp2k-elpa-replay-7632347.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov-1789742762887624799/`, local evidence; not included in Git)
contains `summary.json`, `review.json`, each test's scientific and stderr logs,
the ASE protocol, optimized `atoms.json`/`relax.traj` files, input copies,
GPU health snapshots and MPS logs. The report's core analysis is based on the
local installed source and original core dumps, not a presumed upstream fix.

Two offline regression tests passed: replay variants preserve the saved physical
settings/coordinates, and crash/SCF-failure classification remains distinct.
The PBS script also passed `bash -n` before submission.

## August 12 workflow comparison

The user requested an audit of
`run/parallel-polaris-repo-12Aug26192114-84ec2c/` on September 18.
Its `simulation-results.json` contains **14 successful `run_optimization`
returns and zero failed returns**. The submission/completion events in `run.log`
independently agree: 15 submitted, 14 completed, with `mof-3ecba3bc` still without
a return at the end of the saved log. Its temporary CP2K output ends during
SCF iteration 81. This is a historical incomplete task, not a currently running
job or a recorded ELPA crash. No SIGFPE/ELPA-crash backtrace was found in the
CP2K worker or scientific logs.

The saved configuration launched `deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp`
with eight MPI ranks on two nodes and eight threads per rank. The outputs
identify CP2K 2025.1, revision `9635df4`, compiled August 12, with
`elpa_nvidia_gpu` enabled. The worker script and retained build architecture
both specify CUDA 12.8.1; the worker used GCC 12.3. The current build is CP2K
2025.2 with CUDA 13.0.1 and GCC 14. Both build architectures link NVIDIA-enabled
ELPA 2024.05.001. The old executable is still present.

Every completed August structure requested **197–270 molecular orbitals**.
None exercised the at-most-192-orbital case that leaves an empty column with
four process columns and blocks of 64. The August inputs have no block-size or
ELPA-kernel overrides; the retained CP2K source defaults to blocks of 64 and
selects the NVIDIA GPU kernel for this build. In the same distribution, its
smallest case (197 orbitals) has column widths `64, 64, 64, 5`, while the two
September failures have widths `64, 64, 53, 0` and `64, 64, 64, 0`.

The old and current ELPA source files `src/GPU/CUDA/cuUtils_template.cu` and
`src/elpa2/elpa2_trans_ev_tridi_to_band_template.F90` are byte-for-byte identical
(SHA-256 hashes in the audit JSON). Thus the older source already contains the
same zero-width division. This strengthens the explanation that the August
structures avoided the trigger; it does not demonstrate identical behavior of
the two compiled binaries on an empty-column input. A controlled replay of the
September inputs with the retained August binary has not been performed.

### Completion versus SCF convergence

Each per-MOF `cp2k.out` is a cumulative snapshot of the reused shell's output.
The audit uses only its final force-environment section, avoiding double-counting
earlier MOFs. Each completed task has three SCF evaluations (initial geometry
and two optimization steps). Across the 14 tasks, **10 evaluations converged
and 32 did not**. Eleven tasks have at least one unconverged evaluation; only
three have all three converged. `IGNORE_CONVERGENCE_FAILURE` lets those tasks
return successfully despite reaching the 128-iteration SCF limit.

| MOF | Requested orbitals | SCF outcomes, initial / step 1 / step 2 |
| --- | ---: | --- |
| mof-d2f542fd | 222 | Converged: 39 / 15 / 20 iterations |
| mof-3495dd6e | 235 | Not converged / converged: 71 / not converged |
| mof-93ec9f42 | 226 | All three not converged |
| mof-aed4104a | 197 | All three not converged |
| mof-12655ee0 | 205 | Converged: 41 / 20 / 18 iterations |
| mof-fdb02bfc | 261 | All three not converged |
| mof-96808ff1 | 220 | All three not converged |
| mof-bd49998f | 197 | All three not converged |
| mof-2dce69de | 221 | All three not converged |
| mof-7dc02b17 | 225 | All three not converged |
| mof-e3f280dc | 202 | All three not converged |
| mof-8a765d3b | 270 | Converged: 89 / 36 / 33 iterations |
| mof-bef24fce | 251 | All three not converged |
| mof-7b40d0a4 | 244 | All three not converged |

The user's recollection of no CP2K crashes in this run is supported by the
records. Scientific convergence is a separate limitation of those successful
workflow returns. Per-task source paths, line numbers, output hashes, counts,
and source comparisons are saved in [aug12-audit.json](aug12-audit.json).
