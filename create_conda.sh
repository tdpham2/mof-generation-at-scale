#!/bin/bash

source /lus/eagle/projects/datascience/hari/.local/miniconda3/etc/profile.d/conda.sh
conda activate base
conda env create \
  --file envs/environment-polaris.yml \
  --prefix "$PWD/mofa_env"
conda activate "$PWD/mofa_env"
pip install --no-deps -e .
