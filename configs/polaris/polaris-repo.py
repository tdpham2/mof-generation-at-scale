"""Polaris scaling configuration for a repository-local installation.

All executable and environment paths are derived from this repository, making
the configuration reusable by any user who installs MOFA under ``mofa_env``
and builds LAMMPS and CP2K under ``deps``.
"""

from pathlib import Path

from parsl import Config as ParslConfig
from pydantic import computed_field

from mofa.hpc.config import RASPAVersion, SingleJobHPCConfig


ROOT = Path.cwd().resolve()
if not (ROOT / "run_parallel_workflow.py").is_file():
    raise RuntimeError(
        "Load this configuration from the MOFA repository root. "
        f"Current directory: {ROOT}"
    )


class Config(SingleJobHPCConfig):
    """Use ML-IAP, distributed CP2K, and RASPA2 on Polaris."""

    lammps_cmd: tuple[str, ...] = (
        str(ROOT / "bin/run-lammps-polaris.sh"),
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "on", "neigh", "half",
    )
    raspa_version: RASPAVersion = "raspa2"
    raspa_cmd: tuple[str, ...] = (str(ROOT / "mofa_env/bin/simulate"),)

    worker_init: str = f"""
module reset
module use /soft/modulefiles
module load gcc
module load cudatoolkit-standalone/12.8
module load conda
conda activate base
conda activate "{ROOT}/mofa_env"
export PATH="{ROOT}/mofa_env/bin:${{PATH}}"
export CP2K_DATA_DIR="{ROOT}/deps/cp2k-2025.1/data"
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
runtime_cache="${{TMPDIR:-/tmp}}/mofa-${{PBS_JOBID}}"
mkdir -p "${{runtime_cache}}/matplotlib" "${{runtime_cache}}/xdg"
export MPLCONFIGDIR="${{runtime_cache}}/matplotlib"
export XDG_CACHE_HOME="${{runtime_cache}}/xdg"
cd "{ROOT}"
which python
hostname
""".strip()

    @computed_field
    @property
    def dft_cmd(self) -> str:
        """Launch four GPU-bound CP2K ranks on each assigned node."""
        if self.run_dir is None:
            raise ValueError("run_dir must be set before constructing dft_cmd")
        affinity = ROOT / "bin/set-affinity-gpu-polaris.sh"
        cp2k = ROOT / "deps/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp"
        hostfiles = self.run_dir.absolute() / "cp2k-hostfiles"
        return (
            f"mpiexec -n {self.nodes_per_cp2k * self.gpus_per_node} "
            f"--ppn {self.gpus_per_node} --cpu-bind depth --depth 8 "
            f"-env OMP_NUM_THREADS=8 --hostfile "
            f"{hostfiles}/local_hostfile.`printf %04d $PARSL_WORKER_RANK` "
            f"{affinity} {cp2k}"
        )

    def make_parsl_config(self) -> ParslConfig:
        """Build the standard Polaris layout with this environment on workers."""
        config = super().make_parsl_config()
        for executor in config.executors:
            provider = getattr(executor, "provider", None)
            if provider is not None and hasattr(provider, "worker_init"):
                provider.worker_init = self.worker_init
        return config


hpc_config = Config()
