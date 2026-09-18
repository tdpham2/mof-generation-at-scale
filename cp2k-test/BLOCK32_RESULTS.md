# Standalone CP2K block-size-32 test — 2026-09-18

PBS job **7632443** completed on two Polaris nodes (`x3001c0s19b1n0` and
`x3001c0s1b1n0`), with eight MPI ranks and eight OpenMP threads per rank.
Calculations ran 15:26:28–15:39:12 UTC; PBS walltime was 12 minutes 55 seconds.

**All five CP2K processes exited normally with no ELPA crash. Four inputs
converged their SCF calculation; one reached the 128-iteration limit.**

| Input | CP2K exit code | GPU ELPA / block size | SCF outcome |
| --- | ---: | --- | --- |
| Standalone 112-atom input | 0 | NVIDIA_GPU / 32 | Converged in 44 iterations |
| mof-521f5ced | 0 | NVIDIA_GPU / 32 | Converged in 29 iterations |
| mof-b0c05282 | 0 | NVIDIA_GPU / 32 | Converged in 35 iterations |
| mof-a0d4f10d | 0 | NVIDIA_GPU / 32 | Not converged after 128 iterations |
| mof-155841fd | 0 | NVIDIA_GPU / 32 | Converged in 35 iterations |

The test harness and PBS job returned **exit status 1**, because the harness
requires SCF convergence for every case. This reflects the SCF failure in
`mof-a0d4f10d`, whose last reported convergence error was 0.28943188, rather
than a repeat of the original ELPA divide-by-zero. All outputs contain the
normal CP2K termination marker, and none contain SIGFPE, SIGSEGV, SIGABRT, or
MPI_ABORT. The existing `IGNORE_CONVERGENCE_FAILURE` input setting explains
why CP2K itself returns zero for the unconverged calculation.

## Input changes and scope

The standalone [cp2k.inp](cp2k.inp) sets:

```text
&GLOBAL
  &FM
    NROW_BLOCKS 32
    NCOL_BLOCKS 32
  &END FM
  &PRINT_ELPA ON
  &END PRINT_ELPA
&END GLOBAL
```

Prepared copies of the four failed inputs and the standalone input are in
[block32-inputs](block32-inputs/). Their [manifest](block32-inputs/manifest.json)
records source paths and SHA-256 hashes. Input preparation verified that each
entire FORCE_EVAL section was unchanged, including coordinates, cell, basis,
functional, cutoff, and SCF settings. The actual outputs confirm GPU ELPA and
block size 32 for every case. No production MOFA template or configuration was
modified by this test.

These are standalone energy/force evaluations, not geometry optimizations.
The results support the block-size workaround for these four crash cases;
they do not establish full geometry convergence or eliminate the need to
handle SCF nonconvergence separately.

## Evidence and reproduction

Results: completed run directory (`run/cp2k-block32-7632443.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov-1789745188811845413/`, local evidence; not included in Git).
Its `summary.json` records every launch and validation result; each case has
`cp2k.inp`, `cp2k.out`, `stdout.txt`, `stderr.txt`, and `launch.json`.
For `mof-a0d4f10d`, `cp2k.out:1460` records the SCF nonconvergence warning.

Submit from the repository root:

```bash
qsub cp2k-test/run-block32-polaris.sh
```

The script uses [check-block32.py](check-block32.py), with a ten-minute timeout
per case. Before submission, PBS script syntax, Python compilation, input
preservation checks, and all five existing CP2K diagnostic tests passed.
