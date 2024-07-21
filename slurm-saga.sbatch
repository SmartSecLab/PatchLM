#!/bin/bash
#SBATCH --account=NN9970K
#SBATCH --job-name=test
#SBATCH --partition=a100 --gpus-per-node=1
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCh --cpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=01-00:10:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

source venv/bin/activate
# conda activate patch
which python3
python3 --version

# srun python3 -m src.run
# python3 -m source.run
python3 -m codellama

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`