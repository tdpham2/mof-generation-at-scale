#!/bin/bash -le
#PBS -l select=6:system=polaris
#PBS -l walltime=0:60:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug-scaling
#PBS -N mofa-test
#PBS -A MOFA

hostname

# Change to working directory
cd ${PBS_O_WORKDIR}
pwd

# Activate the environment
conda activate /lus/eagle/projects/MOFA/lward/mof-generation-at-scale/env
which python

# Launch MPS on each node
NNODES=`wc -l < $PBS_NODEFILE`
mpiexec -n ${NNODES} --ppn 1 ./bin/enable_mps_polaris.sh &

# Start Redis
redis-server --bind 0.0.0.0 --appendonly no --logfile redis.log --protected-mode no &
redis_pid=$!
echo launched redis on $redis_pid

# Run
python run_parallel_workflow.py \
      --node-path input-files/zn-paddle-pillar/node.json \
      --ligand-templates input-files/zn-paddle-pillar/template_*_prompt.yml \
      --generator-path models/geom-300k/geom_difflinker_epoch=997_new.ckpt \
      --generator-config-path models/geom-300k/config-tf32-a100.yaml \
      --maximum-train-size 8192 \
      --maximum-strain 0.5 \
      --retrain-freq 1 \
      --num-epochs 128 \
      --num-samples 1024 \
      --gen-batch-size 128 \
      --simulation-budget 32768 \
      --md-timesteps 1000000 \
      --md-snapshots 10 \
      --raspa-timesteps 10000 \
      --lammps-on-ramdisk \
      --dft-opt-steps 2 \
      --dft-fraction 0.25 \
      --ai-fraction 0.4 \
      --proxy-threshold 100000 \
      --compute-config polaris
echo Python done

# Shutdown services
kill $redis_pid
mpiexec -n ${NNODES} --ppn 1 ./bin/disable_mps_polaris.sh
