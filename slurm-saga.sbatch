#!/bin/bash
#SBATCH --account=NN9970K
<<<<<<< HEAD
#SBATCH --job-name=test
=======
#SBATCH --job-name=codellama
>>>>>>> 57126a6a3e1d2150db15c678e7cc5248a44159a9
#SBATCH --partition=a100 --gpus-per-node=1
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCh --cpus-per-task=1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00-00:10:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

<<<<<<< HEAD
source venv/bin/activate
# conda activate patch
which python3
python3 --version

# srun python3 -m src.run
# python3 -m source.run
python3 -m codellama

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`
=======
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
>>>>>>> 57126a6a3e1d2150db15c678e7cc5248a44159a9
