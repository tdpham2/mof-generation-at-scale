"""Test LAMMPS by running a large number of MD simulations with different runtimes"""
from concurrent.futures import as_completed
from platform import node
from pathlib import Path
import argparse
import json

from tqdm import tqdm
from ase import Atoms
import parsl
from parsl.config import Config
from parsl.app.python import PythonApp
from parsl.executors import HighThroughputExecutor
from parsl.providers import PBSProProvider, LocalProvider
from parsl.launchers import MpiExecLauncher

from mofa.model import MOFRecord
from mofa.scoring.geometry import LatticeParameterChange
from mofa.utils.conversions import write_to_string


def test_function(mof: MOFRecord, lammps_invocation: list[str], model_path: str, timesteps: int, device: str = 'cpu') -> tuple[float, list[Atoms]]:
    """Run a MACE-driven LAMMPS MD simulation, report runtime and resultant traj

    MACE's LAMMPS input uses a single ML-IAP/MACE pair_style with no bonded
    (angle/dihedral/improper) terms, so it avoids the UFF4MOF/cif2lammps
    angle-style Kokkos incompatibility that LAMMPSRunner hits (see
    mofa/simulation/cif2lammps/UFF4MOF_construction.py: angle_parameters only
    ever emits cosine/periodic or fourier, neither of which has a LAMMPS
    Kokkos implementation).

    Args:
        mof: MOF to use
        lammps_invocation: Command to invoke LAMMPS
        model_path: Path to the LAMMPS-compatible MACE model
        timesteps: Number of MD time steps
        device: Device used to cache/load the MACE model before the LAMMPS run
    Returns:
        - Runtime (s)
        - MD trajectory
    """
    from mofa.simulation.mace import MACERunner
    from time import perf_counter
    from pathlib import Path

    run_dir = Path(f'run-{timesteps}')
    run_dir.mkdir(exist_ok=True, parents=True)

    # run_molecular_dynamics only invokes run_md_with_lammps, a plain LAMMPS
    # subprocess call with pair_style mliap -- it never touches the ASE
    # mace_mp() calculator, so there's no model to pre-load here. Calling
    # load_model() unconditionally made mace_mp() try to reach the network
    # for the foundation checkpoint, which hung/killed workers on Polaris
    # compute nodes (no outbound internet). See run_parallel_workflow.py,
    # where md_fun (run_molecular_dynamics) likewise never calls load_model().
    runner = MACERunner(run_dir=run_dir, lammps_cmd=lammps_invocation, model_path=Path(model_path), device=device, delete_finished=False)
    start_time = perf_counter()
    output = runner.run_molecular_dynamics(mof, timesteps, timesteps // 5)
    run_time = perf_counter() - start_time

    return run_time, output


if __name__ == "__main__":
    # Get the length of the runs, etc
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', help='Number of timesteps to run', default=1000, type=int)
    parser.add_argument('--config', help='Which compute configuration to use', default='polaris')
    parser.add_argument('--device', help='Which device to use for caching/loading the MACE model', default='cpu')
    args = parser.parse_args()

    # MACE's LAMMPS input uses a single pair_style with no bonded terms
    # (see mofa/simulation/mace.py), so it doesn't hit the UFF4MOF/cif2lammps
    # angle-style Kokkos incompatibility that LAMMPSRunner does.
    model_path = Path('../../input-files/mace/mace-mp0_medium-mliap_lammps.pt').absolute()

    # Select the correct configuraion
    if args.config == "local":
        lammps_cmd = ['/home/lward/Software/lammps-2Aug2023/build/lmp', '-sf', 'omp']
        config = Config(executors=[HighThroughputExecutor(max_workers=1, cpu_affinity='block')])
    elif args.config == "polaris":
        # bin/run-lammps-polaris.sh wraps a build with PKG_KOKKOS=ON and
        # PKG_GPU=OFF (see polaris-build/build-lammps.sh), so it must be
        # invoked with the kokkos suffix/package, not gpu.
        lammps_cmd = (
            '/lus/eagle/projects/datascience/hari/mof-generation-at-scale/bin/run-lammps-polaris.sh '
            '-k on g 1 -sf kk -pk kokkos newton on neigh half'
        ).split()
        config = Config(retries=4, executors=[
            HighThroughputExecutor(
                max_workers_per_node=4,
                cpu_affinity='block-reverse',
                available_accelerators=4,
                provider=LocalProvider(
                    launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1 --no-vni"),
                    #account='MOFA',
                    #queue='debug',
                    #select_options="ngpus=4",
                    #scheduler_options="#PBS -l filesystems=home:eagle",
                    worker_init="""
module list
source /lus/eagle/projects/datascience/hari/mof-generation-at-scale/deps/test/lammps-22Jul2025/venv/bin/activate

# Keep Python multiprocessing sockets below the AF_UNIX path-length limit.
# LocalProvider workers otherwise inherit the login shell's long TMPDIR.
export TMPDIR=/tmp

cd $PBS_O_WORKDIR
pwd
which python
hostname
                    """,
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    #cpus_per_node=32,
                    #walltime="1:00:00",
                )
            )
        ])
    elif args.config == "aurora":
        lammps_cmd = ('/home/lward/MOFA/lward/lammps/lammps-4Feb2025/build-nompi-cpu/lmp',)
        accel_ids = [
            f"{gid}.{tid}"
            for gid in range(6)
            for tid in range(2)
        ]
        config = Config(
            retries=2,
            executors=[
                HighThroughputExecutor(
                    label="sunspot_test",
                    available_accelerators=accel_ids,  # Ensures one worker per accelerator
                    cpu_affinity="block",  # Assigns cpus in sequential order
                    prefetch_capacity=0,
                    max_workers_per_node=12,
                    cores_per_worker=16,
                    provider=PBSProProvider(
                        account="MOFA",
                        queue="debug",
                        worker_init="""
module load frameworks
source /lus/flare/projects/MOFA/lward/mof-generation-at-scale/venv/bin/activate

cd $PBS_O_WORKDIR
pwd
which python
hostname
                        """,
                        walltime="1:00:00",
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind", overrides="--depth=208 --ppn 1"
                        ),  # EnsureDs 1 manger per node and allows it to divide work among all 208 threads
                        scheduler_options="#PBS -l filesystems=home:flare",
                        nodes_per_block=1,
                        min_blocks=0,
                        max_blocks=1,  # Can increase more to have more parallel batch jobs
                        cpus_per_node=208,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f'Configuration not defined: {args.config}')

    # Prepare parsl
    with parsl.load(config):
        test_app = PythonApp(test_function)

        # Submit each MOF
        futures = []
        with open('example-mofs.json') as fp:
            for line in fp:
                mof = MOFRecord(**json.loads(line))
                future = test_app(mof, lammps_cmd, model_path, args.timesteps, args.device)
                future.mof = mof
                futures.append(future)
                break

        # Store results
        # MACERunner.traj_name defaults to 'mace_mp' (mofa/simulation/mace.py),
        # so score against that trajectory key rather than the default 'uff'.
        scorer = LatticeParameterChange(md_level='mace_mp')
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.exception() is not None:
                print(f'{future.mof.name} failed: {future.exception()}')
                continue
            runtime, traj = future.result()

            # Get the strain
            # TODO (wardlt): Simplify how we compute strain
            traj_vasp = [(i, write_to_string(t, 'vasp')) for i, t in traj]
            mof = future.mof
            mof.md_trajectory['mace_mp'] = traj_vasp
            strain = scorer.score_mof(mof)

            # Store the result
            with open('runtimes.json', 'a') as fp:
                print(json.dumps({
                    'host': node(),
                    'lammps_cmd': lammps_cmd,
                    'timesteps': args.timesteps,
                    'mof': mof.name,
                    'runtime': runtime,
                    'strain': strain
                }), file=fp)
