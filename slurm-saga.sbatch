#!/bin/bash
#SBATCH --account=NN9970K
#SBATCH --job-name=test
#SBATCH --partition=accel --gpus-per-node=1
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCh --cpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

echo "Hello world"

python --version
which python

python3 test.py