#!/usr/bin/env python3
"""Run the updated standalone inputs with GPU ELPA on two Polaris nodes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cp2k_diagnostic", ROOT / "cp2k-test/diagnose.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def main():
    job = os.environ.get("PBS_JOBID")
    if not job:
        raise RuntimeError("Submit cp2k-test/run-block32-polaris.sh to PBS")
    hosts = list(dict.fromkeys(Path(os.environ["PBS_NODEFILE"]).read_text().split()))
    if len(hosts) != 2:
        raise RuntimeError("Exactly two allocated nodes required")
    inputs = ROOT / "cp2k-test/block32-inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    output = ROOT / "run" / f"cp2k-block32-{job}-{time.time_ns()}"
    output.mkdir(parents=True)
    hostfile = output / "hosts"
    hostfile.write_text("\n".join(hosts) + "\n")
    shutil.copy2(inputs / "manifest.json", output / "manifest.json")
    env = diag.gpu_diag.clean_environment()
    env.update(OMP_NUM_THREADS="8", MPICH_OFI_CXI_PID_BASE="5")
    interrupted = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: interrupted.__setitem__(0, signum))
    summary = dict(status="running", job=job, hosts=hosts, started_at=diag.gpu_diag.utc(),
                   test="standalone energy/forces; GPU ELPA; block size 32; MPS off", stages=[])

    def save():
        diag.gpu_diag.write_json(output / "summary.json", summary)

    save()
    print(f"Results: {output}", flush=True)
    try:
        for case in manifest["cases"]:
            if interrupted[0]:
                raise RuntimeError("Test interrupted")
            source = inputs / case["input"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != case["sha256"]:
                raise RuntimeError("Input changed after preparation: " + case["name"])
            stage = output / case["name"]
            stage.mkdir()
            shutil.copy2(source, stage / "cp2k.inp")
            argv = ["mpiexec", "-n", "8", "--ppn", "4", "--cpu-bind", "depth", "--depth", "8",
                    "-env", "OMP_NUM_THREADS=8", "-env", "CP2K_BINARY=cp2k.psmp",
                    "--hostfile", str(hostfile), str(ROOT / "bin/set-affinity-gpu-polaris.sh"),
                    str(ROOT / "bin/run-cp2k-polaris.sh"), "-i", "cp2k.inp", "-o", "cp2k.out"]
            command = "cd " + shlex.quote(str(stage)) + " && " + shlex.join(argv)
            print(f"Starting {case['name']}", flush=True)
            result = diag.execute_stage(command, stage, env, 600, interrupted)
            result["name"] = case["name"]
            try:
                if result["status"] != "passed":
                    raise RuntimeError("CP2K launch: " + result["status"])
                result.update(diag.validate_output(stage / "cp2k.out", require_exit=True))
                text = (stage / "cp2k.out").read_text()
                if not re.search(r"ELPA\| Kernel\s+NVIDIA_GPU\b", text):
                    raise RuntimeError("GPU ELPA not confirmed")
                if not re.search(r"ELPA\| Matrix block size \(NBLK\)\s+32\b", text):
                    raise RuntimeError("ELPA block size 32 not confirmed")
                result.update(outcome="scf_converged", elpa_kernel="NVIDIA_GPU", block_size=32)
            except RuntimeError as exc:
                result.update(outcome="failed", validation_error=str(exc))
            summary["stages"].append(result)
            save()
            print(f"{case['name']}: {result['outcome']}", flush=True)
            if result["status"] in ("timeout", "interrupted"):
                raise RuntimeError("Stopping after " + result["status"])
        all_converged = all(s["outcome"] == "scf_converged" for s in summary["stages"])
        summary["status"] = "passed" if all_converged else "failed"
    except Exception as exc:
        summary.update(status="incomplete", error=f"{type(exc).__name__}: {exc}")
    finally:
        summary["finished_at"] = diag.gpu_diag.utc()
        save()
    print(f"Status: {summary['status']}\nResults: {output}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
