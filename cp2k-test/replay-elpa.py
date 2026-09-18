#!/usr/bin/env python3
"""Replay saved workflow inputs to isolate ELPA's empty-column GPU crash."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cp2k_diagnostic", ROOT / "cp2k-test/diagnose.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
write_json = diag.gpu_diag.write_json
CASES = ROOT / "cp2k-test/elpa-cases-18Sep26141032"


def variant_input(text, variant):
    """Change copies only; retain the workflow's SCF and physical settings."""
    from ase.calculators.cp2k import parse_input
    root = parse_input(text)
    if variant == "cpu-elpa":
        root.add_keyword("GLOBAL", "ELPA_KERNEL GENERIC")
    elif variant == "gpu-block32":
        root.add_keyword("GLOBAL/FM", "NROW_BLOCKS 32")
        root.add_keyword("GLOBAL/FM", "NCOL_BLOCKS 32")
    elif variant != "original":
        raise ValueError(variant)
    # Diagnostic output only; no change to numerical settings in 'original'.
    root.get_subsection("GLOBAL/PRINT_ELPA").params = "ON"
    return "\n".join(root.write()) + "\n"


def inspect_output(directory, launch):
    scientific = "\n".join(p.read_text(errors="replace") for p in directory.rglob("cp2k.out"))
    errors = "\n".join(p.read_text(errors="replace") for p in directory.rglob("*stderr*"))
    converged = re.findall(r"SCF run converged in\s+(\d+) steps", scientific)
    unconverged = len(re.findall(r"SCF run NOT converged", scientific, re.I))
    energies = [float(x) for x in re.findall(r"Total energy:\s*([-+\d.Ee]+)", scientific)]
    crash = "SIGFPE" in errors and "unpack_row_group_real_gpu_double" in errors
    status = launch["status"]
    if crash:
        status = "elpa_sigfpe"
    elif status == "passed":
        status = "completed_scf_unconverged" if unconverged else "completed_converged"
        if not converged and not unconverged or not energies or not all(map(math.isfinite, energies)):
            status = "invalid_output"
        if not re.search(r"Total number of message passing processes\s+8\b", scientific):
            status = "invalid_output"
        if not re.search(r"Number of threads for this process\s+8\b", scientific):
            status = "invalid_output"
    return dict(outcome=status, scf_steps=list(map(int, converged)), scf_unconverged=unconverged,
                energies_hartree=energies,
                elpa_details=sorted(set(re.findall(r"^\s*ELPA\|.*$", scientific, re.M))))


def ase_replay(stage, command, cases, variant, steps):
    import numpy as np
    from ase.calculators.cp2k import CP2K
    from ase.optimize import LBFGS

    class SavedInputCP2K(CP2K):
        # ASE still performs LOAD/SET_POS/EVAL_EF/GET_F/GET_STRESS normally.
        # Return the saved input to avoid duplicating coordinates or defaults.
        def _generate_input(self):
            return self.replay_input

    command += " 2> " + shlex.quote(str(stage / "launcher.stderr"))
    metadata = dict(status="running", variant=variant, optimization_steps=steps,
                    set_pos_file=False, cases=[])
    write_json(stage / "ase-result.json", metadata)
    os.chdir(stage)
    calc = None
    try:
        calc = SavedInputCP2K(command=command, debug=True)
        metadata.update(shell_version=calc._shell.version, shell_pid=calc._shell._child.pid)
        for case in cases:
            folder = stage / case["name"]
            folder.mkdir()
            path = CASES / case["input"]
            atoms = diag.input_atoms(path)
            if calc._force_env_id is not None:
                calc._release_force_env()
                calc.reset()
            calc.label = str(folder / "cp2k")
            calc.replay_input = variant_input(path.read_text(), variant)
            atoms.calc = calc
            record = dict(name=case["name"], status="running", evaluations=[])
            metadata["cases"].append(record)
            write_json(stage / "ase-result.json", metadata)

            def capture():
                energy = float(atoms.get_potential_energy())
                forces, stress = atoms.get_forces(), atoms.get_stress()
                if not math.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
                    raise RuntimeError("Non-finite ASE result")
                record["evaluations"].append(dict(energy_eV=energy, max_force=float(np.abs(forces).max()),
                                                  force_shape=list(forces.shape), stress=stress.tolist()))
                write_json(stage / "ase-result.json", metadata)

            with LBFGS(atoms, logfile=str(folder / "relax.log"), trajectory=str(folder / "relax.traj")) as opt:
                opt.attach(capture, interval=1)
                converged = opt.run(fmax=0.01, steps=steps)
            record.update(status="returned", optimizer_converged=bool(converged), steps=opt.nsteps)
            atoms.write(folder / "atoms.json")
            write_json(stage / "ase-result.json", metadata)
        calc.close()
        calc = None
        metadata["status"] = "returned"
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(stage / "ase-result.json", metadata)
        if calc is not None:
            calc.close()


def run(output):
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Submit a two-node PBS compute allocation")
    hosts = list(dict.fromkeys(Path(os.environ["PBS_NODEFILE"]).read_text().split()))
    if len(hosts) != 2:
        raise RuntimeError("Exactly two nodes required")
    manifest = json.loads((CASES / "manifest.json").read_text())
    for case in manifest["cases"]:
        if hashlib.sha256((CASES / case["input"]).read_bytes()).hexdigest() != case["sha256"]:
            raise RuntimeError("Input checksum mismatch: " + case["name"])
    failed = [case for case in manifest["cases"] if case["original_outcome"] == "failed"]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(CASES, output / "original-inputs")
    (output / "cp2k-hostfiles").mkdir()
    hostfile = output / "cp2k-hostfiles/local_hostfile.0000"
    hostfile.write_text("\n".join(hosts) + "\n")
    from mofa.utils.config import load_variable
    config = load_variable(ROOT / "configs/polaris/polaris-repo.py", "hpc_config")
    config.run_dir = output
    if config.nodes_per_cp2k != 2 or config.gpus_per_node != 4:
        raise RuntimeError("Expected two nodes and four ranks per node")
    shell = config.dft_cmd
    direct = shell.replace("CP2K_BINARY=cp2k_shell.psmp", "CP2K_BINARY=cp2k.psmp")
    if direct == shell:
        raise RuntimeError("Cannot derive executable command")
    env = diag.gpu_diag.clean_environment()
    env.update(PARSL_WORKER_RANK="0", OMP_NUM_THREADS="8")
    base = Path("/tmp") / f"mofa-elpa-{time.time_ns()}"
    helper = ["mpiexec", "--no-vni", "-n", "2", "--ppn", "1", "--hostfile", str(hostfile),
              sys.executable, str(ROOT / "cp2k-test/diagnose.py"), "node", "--output", str(output),
              "--mps-base", str(base)]
    summary = dict(status="running", hosts=hosts, started_at=diag.gpu_diag.utc(), stages=[],
                   source_manifest=manifest, shell_command=shell)
    interrupted = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: interrupted.__setitem__(0, signum))
    deadline = time.monotonic() + 54 * 60

    def save():
        write_json(output / "summary.json", summary)

    def nodes(action, label):
        result = diag.execute_stage(helper + ["--action", action, "--label", label],
                                    output / ("node-" + label), env, 90, interrupted)
        if result["status"] != "passed":
            raise RuntimeError(f"Node {action} failed: {label}")

    def stage(cases, variant, interface, mps, steps=0):
        if interrupted[0] or time.monotonic() >= deadline:
            raise RuntimeError("Interrupted or diagnostic execution deadline reached")
        label = f"{len(summary['stages'])+1:02d}-{interface}-{variant}-mps-{'on' if mps else 'off'}-" + "+".join(c["name"] for c in cases)
        folder = output / label
        folder.mkdir()
        if interface == "direct":
            (folder / "cp2k.inp").write_text(variant_input((CASES / cases[0]["input"]).read_text(), variant))
            command = "cd " + shlex.quote(str(folder)) + " && " + direct + " -i cp2k.inp -o cp2k.out"
        else:
            write_json(folder / "request.json", dict(command=shell, cases=cases, variant=variant, steps=steps))
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "ase", "--output", str(folder)]
        print(f"{diag.gpu_diag.utc()} Starting {label}", flush=True)
        result = diag.execute_stage(command, folder, env, min(1500 if steps else 600, deadline-time.monotonic()), interrupted)
        result.update(name=label, variant=variant, interface=interface, mps=mps, cases=[c["name"] for c in cases],
                      **inspect_output(folder, result))
        summary["stages"].append(result)
        save()
        nodes("snapshot", f"stage-{len(summary['stages']):02d}-after")
        print(f"{diag.gpu_diag.utc()} {label}: {result['outcome']}", flush=True)
        if result["status"] in ("timeout", "interrupted"):
            raise RuntimeError("Stage did not terminate normally; stop before more launches")
        return result

    print(f"Results: {output}", flush=True)
    save()
    try:
        nodes("preflight", "before")
        for case in failed:
            stage([case], "original", "direct", False)
        nodes("start", "mps-start")
        env.update(CUDA_MPS_PIPE_DIRECTORY=str(base / "pipes"), CUDA_MPS_LOG_DIRECTORY=str(base / "logs"))
        for case in failed:
            stage([case], "original", "direct", True)
            stage([case], "original", "ase", True)
        controls = {}
        for variant in ("cpu-elpa", "gpu-block32"):
            controls[variant] = [stage([case], variant, "direct", True) for case in failed]
        # Test the workflow's two LBFGS steps on both MOFs in one reused shell.
        preferred = next((variant for variant in ("gpu-block32", "cpu-elpa")
                          if all(r["status"] == "passed" and r["outcome"].startswith("completed_")
                                 for r in controls[variant])), None)
        if preferred:
            stage(failed, preferred, "ase", True, steps=2)
        summary["status"] = "completed"
        summary["note"] = "Expected crash reproductions and SCF validity are recorded separately per stage."
    except Exception as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        interrupted[0] = False
        try:
            nodes("stop", "cleanup")
            summary["cleanup"] = "passed"
        except Exception as exc:
            summary.update(status="incomplete", cleanup_error=str(exc))
        summary["finished_at"] = diag.gpu_diag.utc()
        save()
        print(f"Status: {summary['status']}\nResults: {output}", flush=True)
    return 0 if summary["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "ase"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / "run" / f"cp2k-elpa-replay-{os.environ.get('PBS_JOBID', 'local')}-{time.time_ns()}").resolve()
    if args.mode == "ase":
        request = json.loads((output / "request.json").read_text())
        ase_replay(output, **request)
        return 0
    return run(output)


if __name__ == "__main__":
    sys.exit(main())
