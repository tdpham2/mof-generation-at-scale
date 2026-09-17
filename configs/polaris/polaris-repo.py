"""Polaris scaling configuration for a repository-local installation.

All executable and environment paths are derived from this repository, making
the configuration reusable by any user who installs MOFA under ``mofa_env``
and builds LAMMPS and CP2K under ``deps``.
"""

from pathlib import Path
from subprocess import Popen

from parsl import Config as ParslConfig
from parsl.launchers import WrappedLauncher
from pydantic import Field, computed_field

from mofa.hpc.config import RASPAVersion, SingleJobHPCConfig


ROOT = Path.cwd().resolve()
if not (ROOT / "run_parallel_workflow.py").is_file():
    raise RuntimeError(
        "Load this configuration from the MOFA repository root. "
        f"Current directory: {ROOT}"
    )


# These executors use mpiexec only to place independent Parsl managers on each
# node. Skip Slingshot VNI allocation so they do not consume CXI resources.
NO_VNI_EXECUTORS = {"inf", "train", "lammps", "helper"}
CP2K_CXI_PID_BASE = 5


class Config(SingleJobHPCConfig):
    """Use ML-IAP, distributed CP2K, and RASPA2 on Polaris."""

    lammps_per_gpu: int = Field(default=2, init=False)
    # Run each CP2K job on a single node (-n 4 --ppn 4), matching the layout
    # proven by cp2k-test/run-polaris.sh. The 2-node distributed path is
    # validated separately by cp2k-test/run-polaris-2node.sh.
    nodes_per_cp2k: int = Field(default=2, init=False)

    lammps_cmd: tuple[str, ...] = (
        str(ROOT / "bin/run-lammps-polaris.sh"),
        "-k", "on", "g", "1",
        "-sf", "kk",
        "-pk", "kokkos", "newton", "on", "neigh", "half",
    )
    raspa_version: RASPAVersion = "raspa2"
    raspa_cmd: tuple[str, ...] = (str(ROOT / "mofa_env/bin/simulate"),)

    # The site `conda` module's collection has been broken (all its modules
    # were removed), so this sources the local miniconda from ~/.bashrc
    # directly instead of `module load conda`. See build-cp2k.sh's local
    # toolchain note.
    #
    # mofa_env's compiled deps (raspa2, chargemol, torch, cupy) resolve their
    # own libraries (bundled nvidia-*-cu12 pip packages, or shared libs copied
    # into mofa_env/lib at conda-build time) with no module loaded at all, so
    # this does not load a `gcc` or `cudatoolkit-standalone` module -- doing
    # so is not just unnecessary, `module load gcc` actively breaks mpi4py by
    # deactivating cray-mpich (Lmod prints "Inactive Modules: cray-libsci,
    # cray-mpich" when this happens). The one thing mofa_env's mpi4py does
    # need from a module is the standard-ABI MPI shim, which the default
    # PrgEnv-nvidia stack does not provide on its own -- cray-mpich only ships
    # libmpi_nvidia.so.12, not the libmpi.so.12 mpi4py's wheel expects -- so
    # cray-mpich-abi is loaded in place of it.
    worker_init: str = f"""
module use /soft/modulefiles
module reset
module use /soft/modulefiles
module load cray-mpich-abi
source "/lus/eagle/projects/datascience/hari/.local/miniconda3/etc/profile.d/conda.sh"
conda activate "{ROOT}/mofa_env"
export PATH="{ROOT}/mofa_env/bin:${{PATH}}"
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
        # run-cp2k-polaris.sh loads the module stack CP2K was built against
        # (see build-cp2k.sh) fresh for each rank, the same way lammps_cmd
        # routes through run-lammps-polaris.sh. worker_init only activates
        # mofa_env.
        cp2k = ROOT / "bin/run-cp2k-polaris.sh"
        hostfiles = self.run_dir.absolute() / "cp2k-hostfiles"
        return (
            f"env MPICH_OFI_CXI_PID_BASE={CP2K_CXI_PID_BASE} "
            f"mpiexec -n {self.nodes_per_cp2k * self.gpus_per_node} "
            f"--ppn {self.gpus_per_node} --cpu-bind depth --depth 8 "
            f"-env OMP_NUM_THREADS=8 --hostfile "
            f"{hostfiles}/local_hostfile.`printf %04d $PARSL_WORKER_RANK` "
            f"{affinity} {cp2k}"
        )

    def launch_monitor_process(self, freq: int = 20) -> Popen:
        """Launch one independent utilization monitor per node without a VNI."""
        log_dir = self.run_dir / "logs"
        return Popen(
            args=(
                f"mpiexec --no-vni -n {len(self.hosts)} --ppn 1 "
                f"--depth={self.cpus_per_node} --cpu-bind depth "
                f"monitor_utilization --frequency {freq} {log_dir.absolute()}"
            ).split()
        )

    def make_parsl_config(self) -> ParslConfig:
        """Build the standard Polaris layout with this environment on workers."""
        config = super().make_parsl_config()
        configured_launchers = set()
        for executor in config.executors:
            provider = getattr(executor, "provider", None)
            if provider is not None and hasattr(provider, "worker_init"):
                provider.worker_init = self.worker_init

            if executor.label not in NO_VNI_EXECUTORS:
                continue
            launcher = getattr(provider, "launcher", None)
            if not isinstance(launcher, WrappedLauncher):
                raise TypeError(
                    f"Executor {executor.label!r} must use WrappedLauncher "
                    "to configure its Polaris VNI behavior"
                )
            if not launcher.prepend.startswith("mpiexec "):
                raise ValueError(
                    f"Executor {executor.label!r} has an unexpected launcher: "
                    f"{launcher.prepend!r}"
                )
            launcher.prepend = launcher.prepend.replace(
                "mpiexec ", "mpiexec --no-vni ", 1
            )
            configured_launchers.add(executor.label)

        missing = NO_VNI_EXECUTORS - configured_launchers
        if missing:
            raise RuntimeError(
                "Missing Polaris executors requiring --no-vni: "
                + ", ".join(sorted(missing))
            )
        return config


hpc_config = Config()
