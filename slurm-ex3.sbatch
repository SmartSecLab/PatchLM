#!/bin/bash
#SBATCH -p hgx2q
#SBATCH  --job-name=codellama
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --time 05-00:00:00    # time (D-HH:MM:SS)
#SBATCH --output=output/slurm%j.out         
#SBATCH --error=output/slurm%j.err       

ulimit -s 10240
echo "Job started at:" `date +"%Y-%m-%d %H:%M:%S"`

module purge

source venv/bin/activate
# conda activate patch
which python3
python3 --version

# srun python3 -m src.run
# python3 -m source.run
# python3 -m codellama
python3 -m source.run_codellama

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`