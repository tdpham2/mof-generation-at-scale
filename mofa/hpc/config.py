"""Configuring a particular HPC resource"""
from dataclasses import dataclass, field
from functools import cached_property
from subprocess import Popen
from typing import Literal
from pathlib import Path
from math import ceil
import os

from more_itertools import batched

from parsl import HighThroughputExecutor
from parsl import Config
from parsl.launchers import WrappedLauncher, SimpleLauncher
from parsl.providers import LocalProvider


class HPCConfig:
    """Base class for HPC configuration"""

    # How tasks run
    torch_device: str = 'cpu'
    """Device used for DiffLinker training"""
    lammps_cmd: tuple[str] = ('lmp_serial',)
    """Command used to launch a non-MPI LAMMPS task"""
    cp2k_cmd: str = 'cp2k_shell.psmp'
    """Command used to launch the CP2K shell"""
    graspa_cmd: tuple[str] = ('/home/lward/Software/gRASPA/graspa-sycl/bin/sycl.out',)
    """Command used to launch gRASPA-sycl"""
    lammps_env: dict[str, str] = field(default_factory=dict)
    """Extra environment variables to include when running LAMMPS"""

    # Settings related to distributed training
    gpus_per_node: int = 1
    """How many GPUs per compute node"""
    num_training_nodes: int = 1
    """How many nodes to use for training operations"""
    training_nodes: list[str] = ()
    """Names of nodes participating in training"""

    # How tasks are distributed
    ai_fraction: float = 0.1
    """Maximum fraction of resources set aside for AI tasks"""
    dft_fraction: float = 0.4
    """Maximum fraction of resources not used for AI that will be used for CP2K"""
    lammps_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for simulation tasks"""
    cp2k_executors: Literal['all'] | list[str] = 'all'
    """Which executors to use for CP2K tasks"""
    raspa_executors: Literal['all'] | list[str] = 'all'
    """Which executors to use for RASPA tasks"""
    inference_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for AI tasks"""
    train_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for AI tasks"""
    helper_executors: Literal['all'] | list[str] = 'all'
    """Which executors are available for processing tasks"""

    @property
    def num_training_ranks(self):
        return self.gpus_per_node * self.num_training_nodes

    @property
    def num_workers(self) -> int:
        """Total number of workers"""
        return self.num_lammps_workers + self.num_cp2k_workers + self.number_inf_workers

    @property
    def number_inf_workers(self) -> int:
        """Number of workers set aside for AI inference tasks"""
        raise NotImplementedError

    @property
    def num_lammps_workers(self) -> int:
        """Number of workers available for LAMMPS tasks"""
        raise NotImplementedError

    @property
    def num_cp2k_workers(self):
        """Number of workers available for CP2K tasks"""
        raise NotImplementedError

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        """Launch a monitor process on all resources

        Args:
            log_dir: Folder in which to save logs
            freq: Interval between monitoring
        Returns:
            Process handle
        """
        raise NotImplementedError

    def make_parsl_config(self, run_dir: Path) -> Config:
        """Make a Parsl configuration

        Args:
            run_dir: Directory in which results will be stored
        Returns:
            Configuration that saves Parsl logs into the run directory
        """
        raise NotImplementedError()


@dataclass(kw_only=True)
class LocalConfig(HPCConfig):
    """Configuration used for testing purposes. Runs all non-helper tasks on a single worker
    """

    torch_device = 'cuda'
    lammps_env = {}
    lammps_cmd = ('/home/lward/Software/lammps-mace/build-mace/lmp',)
    cp2k_cmd = ('/home/lward/Software/cp2k-2024.2/exe/local_cuda/cp2k_shell.ssmp',)
    graspa_cmd = ('/home/lward/Software/gRASPA/graspa-sycl/bin/sycl.out,')

    lammps_executors = ['gpu']
    inference_executors = ['gpu']
    train_executors = ['gpu']
    helper_executors = ['helper']
    raspa_executors = ['gpu']

    @property
    def num_workers(self):
        return self.num_lammps_workers + self.num_cp2k_workers + self.number_inf_workers

    @property
    def number_inf_workers(self) -> int:
        return 1

    @property
    def num_lammps_workers(self) -> int:
        return 1

    @property
    def num_cp2k_workers(self) -> int:
        return 1

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='gpu', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class LocalXYConfig(HPCConfig):
    """Configuration Xiaoli uses for testing purposes"""

    torch_device = 'cuda'
    lammps_cmd = "/home/xyan11/software/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1".split()
    lammps_env = {}
    cp2k_cmd = "OMP_NUM_THREADS=1 mpiexec -np 8 /home/xyan11/software/cp2k-v2024.1/exe/local/cp2k_shell.psmp"
    lammps_executors = ['sim']
    inference_executors = ['ai']
    train_executors = ['ai']
    helper_executors = ['helper']

    @property
    def num_workers(self):
        return self.num_lammps_workers + self.num_cp2k_workers + self.number_inf_workers

    @property
    def number_inf_workers(self) -> int:
        return 1

    @property
    def num_lammps_workers(self) -> int:
        return 1

    @property
    def num_cp2k_workers(self) -> int:
        return 1

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='sim', max_workers_per_node=1),
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='ai', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class UICXYConfig(HPCConfig):
    """Configuration Xiaoli uses for uic hpc"""

    torch_device = 'cuda'
    lammps_cmd = "/projects/cme_santc/xyan11/software/source/lmp20230802up3/build-gpu/lmp -sf gpu -pk gpu 1".split()
    lammps_env = {}

    lammps_executors = ['sim']
    ai_executors = ['ai']
    helper_executors = ['helper']

    cp2k_cmd = ("OMP_NUM_THREADS=2 mpirun -np 4 singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd "
                "/projects/cme_santc/xyan11/software/source/cp2k_v2023.1.sif cp2k_shell.psmp")

    @property
    def num_workers(self):
        return self.num_lammps_workers + self.num_cp2k_workers + self.num_ai_workers

    @property
    def num_ai_workers(self) -> int:
        return 2

    @property
    def num_lammps_workers(self) -> int:
        return 3

    @property
    def num_cp2k_workers(self) -> int:
        return 1

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f"monitor_utilization --frequency {freq} {log_dir}".split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        return Config(
            executors=[
                HighThroughputExecutor(label='sim', max_workers_per_node=4, available_accelerators=4),
                HighThroughputExecutor(label='helper', max_workers_per_node=1),
                HighThroughputExecutor(label='ai', max_workers_per_node=1, available_accelerators=1)
            ],
            run_dir=str(run_dir / 'runinfo')
        )


@dataclass(kw_only=True)
class PolarisConfig(HPCConfig):
    """Configuration used on Polaris"""

    torch_device = 'cuda'
    lammps_cmd = ('/lus/eagle/projects/MOFA/lward/lammps-29Aug2024/build-gpu-nompi-mixed/lmp '
                  '-sf gpu -pk gpu 1').split()
    lammps_env = {}
    run_dir: str | None = None  # Set when building the configuration

    nodes_per_cp2k: int = 2
    """Number of nodes per CP2K task"""
    lammps_per_gpu: int = field(default=4, init=False)
    """Number of LAMMPS to run per GPU"""

    ai_hosts: list[str] = field(default_factory=list)
    """Hosts which will run AI tasks"""
    lammps_hosts: list[str] = field(default_factory=list)
    """Hosts which will run LAMMPS tasks"""
    cp2k_hosts: list[str] = field(default_factory=list)
    """Hosts which will run CP2K tasks"""

    cpus_per_node: int = field(default=32, init=False)
    """Number of CPUs to use per node"""
    gpus_per_node: int = field(default=4, init=False)
    """Number of GPUs per compute node"""

    lammps_executors = ['lammps']
    inference_executors = ['inf']
    train_executors = ['train']
    cp2k_executors = ['cp2k']
    helper_executors = ['helper']
    raspa_executors = ['lammps']

    @property
    def cp2k_cmd(self):
        # TODO (wardlt): Turn these into factory classes to ensure everything gets set on build
        assert self.run_dir is not None, 'This must be run after the Parsl config is built'
        return (f'mpiexec -n {self.nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                f'--hostfile {self.run_dir}/cp2k-hostfiles/local_hostfile.`printf %03d $PARSL_WORKER_RANK` '
                '/lus/eagle/projects/MOFA/lward/cp2k-2025.1/set_affinity_gpu_polaris.sh '
                '/lus/eagle/projects/MOFA/lward/cp2k-2025.1/exe/local_cuda/cp2k_shell.psmp')

    @cached_property
    def hosts(self):
        """Lists of hosts on which this computation is running"""
        # Determine the number of nodes from the PBS_NODEFILE
        node_file = os.environ['PBS_NODEFILE']
        with open(node_file) as fp:
            hosts = [x.strip() for x in fp]

        # Determine the number of nodes to use for AI
        num_ai_hosts = max(self.num_training_nodes, min(int(self.ai_fraction * len(hosts)), len(hosts) - self.nodes_per_cp2k - 1))
        self.ai_hosts = hosts[:num_ai_hosts]
        if num_ai_hosts < self.num_training_nodes:
            raise ValueError(f'We need at least {self.num_training_nodes} AI workers. Increase node count or ai_fraction')

        # Determine the number of hosts to use for simulation
        sim_hosts = hosts[num_ai_hosts:]
        max_cp2k_slots = len(sim_hosts) // self.nodes_per_cp2k
        num_cp2k_slots = max(1, min(int(self.dft_fraction * max_cp2k_slots), max_cp2k_slots))  # [nodes_per_cp2k, len(sim_hosts) - nodes_per_cp2k]
        num_dft_hosts = num_cp2k_slots * self.nodes_per_cp2k
        self.lammps_hosts = sim_hosts[num_dft_hosts:]
        self.cp2k_hosts = sim_hosts[:num_dft_hosts]
        return hosts

    @property
    def number_inf_workers(self):
        return (len(self.ai_hosts) - self.num_training_nodes) * self.gpus_per_node

    @property
    def num_lammps_workers(self):
        return len(self.lammps_hosts) * self.gpus_per_node * self.lammps_per_gpu

    @property
    def num_cp2k_workers(self):
        return ceil(len(self.cp2k_hosts) / self.nodes_per_cp2k)

    def launch_monitor_process(self, log_dir: Path, freq: int = 20) -> Popen:
        return Popen(
            args=f'mpiexec -n {len(self.hosts)} --ppn 1 --depth={self.cpus_per_node} '
                 f'--cpu-bind depth monitor_utilization --frequency {freq} {log_dir.absolute()}'.split()
        )

    def make_parsl_config(self, run_dir: Path) -> Config:
        self.run_dir = str(run_dir.absolute())  # Used for CP2K config
        assert len(self.hosts) > 0, 'No hosts detected'

        # Write the nodefiles
        ai_nodefile, lammps_nodefile = self._make_nodefiles(run_dir)

        # Use the same worker_init for most workers
        worker_init = """
module use /soft/modulefiles
module list
source /home/lward/miniconda3/bin/activate /lus/eagle/projects/MOFA/lward/mof-generation-at-scale/env
which python
hostname""".strip()

        # Divide CPUs on "sim" such that a from each NUMA affinity are set aside for helpers
        #  See https://docs.alcf.anl.gov/polaris/hardware-overview/machine-overview/#polaris-device-affinity-information
        sim_cores, helper_cores = self._assign_cores()
        sim_cores.reverse()

        cpus_per_worker = self.cpus_per_node // self.gpus_per_node
        ai_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - 1}" for i in range(4)][::-1]  # All CPUs to AI tasks

        # Launch 4 workers per node, one per GPU
        return Config(executors=[
            HighThroughputExecutor(
                label='inf',
                max_workers_per_node=4,
                cpu_affinity='list:' + ":".join(ai_cores),
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.ai_hosts) - self.num_training_nodes} --ppn 1 --hostfile {ai_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='train',
                max_workers_per_node=1,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n 1 --ppn 1 --host {self.ai_hosts[0]} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='lammps',
                max_workers_per_node=self.lammps_per_gpu * self.gpus_per_node,
                cpu_affinity='list:' + ":".join(sim_cores),
                available_accelerators=self.lammps_per_gpu * self.gpus_per_node,
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.lammps_hosts)} --ppn 1 --hostfile {lammps_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='cp2k',
                max_workers_per_node=self.num_cp2k_workers,
                cores_per_worker=1e-6,
                provider=LocalProvider(
                    launcher=SimpleLauncher(),  # Places a single worker on the launch node
                    min_blocks=1,
                    max_blocks=1
                )
            ),
            HighThroughputExecutor(
                label='helper',
                max_workers_per_node=len(helper_cores),
                cpu_affinity='list:' + ":".join(helper_cores),
                provider=LocalProvider(
                    launcher=WrappedLauncher(
                        f"mpiexec -n {len(self.lammps_hosts)} --ppn 1 --hostfile {lammps_nodefile} --depth=64 --cpu-bind depth"
                    ),
                    worker_init=worker_init,
                    min_blocks=1,
                    max_blocks=1
                )
            ),
        ],
            run_dir=str(run_dir),
            usage_tracking=3,
        )

    def _assign_cores(self):
        """Assign cores on nodes running LAMMPS to both LAMMPS and helper functions

        Returns:
            - List of cores to use for each LAMMPS worker
            - List of cores to use for each helper worker
        """

        lammps_per_node = self.gpus_per_node * self.lammps_per_gpu
        cpus_per_worker = self.cpus_per_node // lammps_per_node
        helpers_per_worker = 1  # One core per worker set aside for "helpers"
        sim_cores = [f"{i * cpus_per_worker}-{(i + 1) * cpus_per_worker - helpers_per_worker - 1}" for i in range(lammps_per_node)]
        helper_cores = [str(i) for w in range(lammps_per_node) for i in range((w + 1) * cpus_per_worker - helpers_per_worker, (w + 1) * cpus_per_worker)]
        return sim_cores, helper_cores

    def _make_nodefiles(self, run_dir: Path):
        """Write the nodefiles for each type of workers to disk

        Run after setting the run directory

        Writes nodefiles for the AI and LAMMPS tasks,
        and a directory of nodefiles to be used for each CP2K instance

        Args:
            run_dir: Run directory for the computation
        Returns:
            - Path to the AI nodefile
            - Path to the LAMMPS nodefile
        """
        assert len(self.hosts) > 0, 'No hosts detected'  # TODO (wardlt): Also builds the hosts list, make that clearer/auto

        ai_nodefile = run_dir / 'ai.hosts'
        ai_nodefile.write_text('\n'.join(self.ai_hosts[self.num_training_nodes:]))  # First are used for training
        lammps_nodefile = run_dir / 'lammps.hosts'
        lammps_nodefile.write_text('\n'.join(self.lammps_hosts) + '\n')
        cp2k_nodefile = run_dir / 'cp2k.hosts'
        cp2k_nodefile.write_text('\n'.join(self.cp2k_hosts) + '\n')

        # Make the nodefiles for the CP2K workers
        nodefile_path = run_dir / 'cp2k-hostfiles'
        nodefile_path.mkdir(parents=True)
        for i, nodes in enumerate(batched(self.cp2k_hosts, self.nodes_per_cp2k)):
            (nodefile_path / f'local_hostfile.{i:03d}').write_text("\n".join(nodes))
        return ai_nodefile, lammps_nodefile


@dataclass(kw_only=True)
class AuroraConfig(PolarisConfig):
    """Configuration for running on Sunspot

    Each GPU tasks uses a single tile"""

    torch_device = 'xpu'
    lammps_cmd = (
        "/lus/flare/projects/MOFA/lward/lammps-kokkos/src/lmp_macesunspotkokkos "
        "-k on g 1 -sf kk"
    ).split()
    lammps_env = {'OMP_NUM_THREADS': '1'}
    graspa_cmd = (
        '/lus/flare/projects/MOFA/lward/gRASPA/graspa-sycl/bin/sycl.out'
    ).split()
    cpus_per_node = 96
    gpus_per_node = 12
    lammps_per_gpu = 1
    max_helper_nodes = 256

    worker_init = """
# General environment variables
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
conda deactivate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Needed for LAMMPS
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH
    """.strip()

    @property
    def training_nodes(self):
        return self.ai_hosts[:self.num_training_nodes]

    @property
    def cp2k_cmd(self):
        assert self.run_dir is not None, 'This must be run after the Parsl config is built'
        return (f'mpiexec -n {self.nodes_per_cp2k * self.gpus_per_node} --ppn {self.gpus_per_node}'
                f' --cpu-bind depth --depth={104 // self.gpus_per_node} -env OMP_NUM_THREADS={104 // self.gpus_per_node} '
                '--env OMP_PLACES=cores '
                f'--hostfile {self.run_dir}/cp2k-hostfiles/local_hostfile.`printf %03d $PARSL_WORKER_RANK` '
                '/lus/flare/projects/MOFA/lward/mof-generation-at-scale/bin/cp2k_shell')

    def make_parsl_config(self, run_dir: Path) -> Config:
        assert self.num_training_nodes == 1, 'Only supporting a single training node for now'
        # Set the run dir and write nodefiles to it
        self.run_dir = str(run_dir.absolute())
        ai_nodefile, lammps_nodefile = self._make_nodefiles(run_dir)

        # Make a helper node file from a subset of lammps nodes
        helper_nodefile = run_dir / 'helper.nodes'
        helper_nodefile.write_text("\n".join(self.lammps_hosts[:self.max_helper_nodes]) + "\n")

        # Determine which cores to use for AI tasks
        sim_cores, helper_cores = self._assign_cores()

        worker_init = """
# General environment variables
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Needed for LAMMPS
FPATH=/opt/aurora/24.180.3/frameworks/aurora_nre_models_frameworks-2024.2.1_u1/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$FPATH/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FPATH/intel_extension_for_pytorch/lib:$LD_LIBRARY_PATH

# Put RASPA2 on the path
export PATH=$PATH:`realpath conda-env/bin/`

cd $PBS_O_WORKDIR
pwd
which python
hostname"""

        return Config(
            executors=[
                HighThroughputExecutor(
                    label='inf',
                    max_workers_per_node=12,
                    cpu_affinity="block",
                    available_accelerators=12,
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            f"mpiexec -n {len(self.ai_hosts) - self.num_training_nodes} --ppn 1 --hostfile {ai_nodefile} --depth=104 --cpu-bind depth"
                        ),
                        worker_init=worker_init,
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
                HighThroughputExecutor(
                    label='train',
                    max_workers_per_node=12,
                    available_accelerators=12,
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            f"mpiexec -n 1 --ppn 1 --host {self.ai_hosts[0]} --depth=104 --cpu-bind depth"
                        ),
                        worker_init=worker_init,
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
                HighThroughputExecutor(
                    label="lammps",
                    available_accelerators=12,
                    cpu_affinity='list:' + ":".join(sim_cores),
                    prefetch_capacity=0,
                    max_workers_per_node=12,
                    provider=LocalProvider(
                        worker_init=worker_init,
                        launcher=WrappedLauncher(
                            f"mpiexec -n {len(self.lammps_hosts)} --ppn 1 --hostfile {lammps_nodefile} --depth=104 --cpu-bind depth"
                        ),
                        min_blocks=1,
                        max_blocks=1,
                    ),
                ),
                HighThroughputExecutor(
                    label='cp2k',
                    max_workers_per_node=self.num_cp2k_workers,
                    cores_per_worker=1e-6,
                    provider=LocalProvider(
                        launcher=SimpleLauncher(),  # Places a single worker on the launch node
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
                HighThroughputExecutor(
                    label='helper',
                    max_workers_per_node=len(helper_cores),
                    cpu_affinity='list:' + ":".join(helper_cores),
                    provider=LocalProvider(
                        launcher=WrappedLauncher(
                            f"./envs/aurora/parallel.sh {helper_nodefile}"
                        ),
                        worker_init=worker_init,
                        min_blocks=1,
                        max_blocks=1
                    )
                ),
            ],
            run_dir=str(run_dir)
        )


configs: dict[str, type[HPCConfig]] = {
    'local': LocalConfig,
    'localXY': LocalXYConfig,
    'UICXY': UICXYConfig,
    'polaris': PolarisConfig,
    'aurora': AuroraConfig
}
