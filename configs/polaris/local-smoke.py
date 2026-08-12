"""Single-node Polaris configuration for an end-to-end MOFA smoke test."""

from pathlib import Path

from pydantic import computed_field

from mofa.hpc.config import LocalConfig, RASPAVersion


ROOT = Path.cwd().resolve()
if not (ROOT / "run_parallel_workflow.py").is_file():
    raise RuntimeError(
        "Load this configuration from the MOFA repository root. "
        f"Current directory: {ROOT}"
    )


class Config(LocalConfig):
    """Run each accelerator-backed stage sequentially on one Polaris GPU."""

    lammps_cmd: tuple[str, ...] = (
        str(ROOT / "bin/run-lammps-polaris.sh"),
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "on", "neigh", "half",
    )

    raspa_version: RASPAVersion = "raspa2"
    raspa_cmd: tuple[str, ...] = (
        str(ROOT / "mofa_env/bin/simulate"),
    )

    # LocalConfig otherwise permits DFT on the CPU helper executor as well.
    dft_executors: list[str] = ["gpu"]
    raspa_executors: list[str] = ["gpu"]

    @computed_field
    @property
    def dft_cmd(self) -> str:
        return str(
            ROOT
            / "deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.ssmp"
        )


hpc_config = Config()
