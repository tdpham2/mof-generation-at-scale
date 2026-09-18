# Two-node CP2K executable and ASE-shell diagnostic

The completed 2026-09-18 run passed all four stages in job `7632220`.
See [results and evidence](RESULTS.md).

The follow-up for the generated MOFs that crash in ELPA is documented in
[ELPA_REPLAY.md](ELPA_REPLAY.md), with a separate replay script and saved inputs.

## Standalone block-size-32 input check

Job **7632443** completed: all five inputs avoided the ELPA crash, four converged
SCF, and `mof-a0d4f10d` reached its SCF iteration limit. See [results](BLOCK32_RESULTS.md).

`cp2k.inp` now sets `GLOBAL/FM/NROW_BLOCKS 32` and `NCOL_BLOCKS 32`, with
`PRINT_ELPA ON` to verify the actual solver configuration. Prepared inputs in
`block32-inputs/` include this 112-atom case and the four failed workflow
structures (`mof-521f5ced`, `mof-b0c05282`, `mof-a0d4f10d`, `mof-155841fd`).
The manifest records source paths and hashes; the original failed inputs remain
available for comparison. Only the GLOBAL settings differ from their sources.

```bash
qsub cp2k-test/run-block32-polaris.sh
```

This runs five standalone CP2K energy/force calculations on two nodes, with
eight MPI ranks, eight threads per rank, and MPS off. It checks normal exit,
SCF convergence, GPU execution, `NVIDIA_GPU` ELPA, and block size 32. Results
and a `summary.json` are saved under `run/cp2k-block32-<job>-<timestamp>/`.
Each case has a ten-minute timeout. These calculations test the eigensolver
workaround; they do not establish geometry convergence. The MOFA input template
and production workflow settings are separate from these test inputs.

## Executable and ASE-shell diagnostic

Submit from the repository root:

```bash
qsub cp2k-test/run-diagnostic-polaris.sh
```

This requests two Polaris nodes for one hour in `debug`, account `ChemGraph`.
It activates `mofa_env_py312` unless `MOFA_ENV` is explicitly supplied, then uses
the **actual `dft_cmd` from `configs/polaris/polaris-repo.py`**: eight MPI ranks,
four ranks per node, eight OpenMP threads, GPU affinity wrapper and CXI PID base 5.
Only independent node-health helpers use `mpiexec --no-vni`.

The four stages are:

1. `cp2k.psmp`, MPS off.
2. ASE using `cp2k_shell.psmp`, MPS off.
3. `cp2k.psmp`, MPS on.
4. ASE using `cp2k_shell.psmp`, MPS on.

All stages use the 112-atom structure in `cp2k-test/cp2k.inp`. Direct runs use a
copy of that input. ASE uses MOFA's CP2K template, basis, pseudopotential, cutoff,
and SCF limit. Both require SCF convergence. Each ASE stage checks energy, forces,
and stress, then displaces one atom by 0.001 Angstrom and evaluates again through
the **same shell process**. This exercises the current inline coordinate-transfer
protocol, including ASE's warning for MPI structures over 100 atoms. It does not
silently switch to `set_pos_file=True` or claim a geometry optimization converged.

Every GPU on both nodes is checked before calculations and after each stage.
Existing MPS daemons prevent startup and are never stopped. MPS-on stages use
private directories and preserve per-node daemon/server logs. Each calculation
stage has a 20-minute timeout; the entire diagnostic stops launching work at its
55-minute deadline and cleans up its own MPS daemons. An unsuccessful stage stops
later stages so it can be diagnosed before adding another variable.

Results appear under `run/cp2k-diagnostic-<job>-<timestamp>/`:

- `summary.json`: commands, hosts, stage outcomes and timings.
- `<stage>/cp2k.out`: CP2K's scientific output.
- `<stage>/stdout.txt`: launcher output, or ASE's detailed shell protocol.
- `<stage>/stderr.txt`: launcher/Python stderr.
- `<ASE stage>/launcher.stderr`: the actual MPI/CP2K child stderr. This is where
  CUDA initialization, MPI aborts, and other causes behind an ASE assertion appear.
- `<ASE stage>/ase-result.json`: shell version, both evaluations, forces and errors.
- `nodes/<hostname>/`: GPU health, MPS startup/shutdown evidence and archived logs.

Acceptance checks eight ranks, eight threads, accelerator use, converged
SCF and finite outputs. Direct runs require the DBCSR device-count banner;
CP2K 2025.2 shell mode omits this banner, so ASE runs can instead demonstrate
GPU use through executed GPU FFT routines in the final timing table. The
summary records which evidence was available. This validates CP2K's launch and ASE interface, not Parsl
scheduling, every generated MOF, or the health of previously allocated nodes.

Offline checks (no GPU or PBS job required):

```bash
PYTHONNOUSERSITE=1 mofa_env_py312/bin/python -m unittest discover \
  -s tests -p test_cp2k_diagnostic.py
```
