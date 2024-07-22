#!/bin/bash
#SBATCH --account=NN9970K
#SBATCH --job-name=codellama
#SBATCH --partition=a100 --gpus-per-node=1
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCh --cpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00-00:10:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

export HF_HOME=/cluster/projects/nn9970k/codellama
echo "Hello world"

# python3 -m venv venv
source venv/bin/activate
# python3 -m pip install --upgrade pip


# python3 -m pip install -r requirements.txt
# python3 -m pip install peft

python --version
which python


python3 -m codellama