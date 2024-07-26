#!/bin/bash
#SBATCH -p hgx2q
#SBATCH  --job-name=codellama
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --time 01-00:00:00    # time (D-HH:MM:SS)
#SBATCH --output=output/slurm%j.out         # Output file name (%j expands to jobID)
#SBATCH --error=output/slurm%j.err          # Error file name (%j expands to jobID)

ulimit -s 10240
echo "Job started at:" `date +"%Y-%m-%d %H:%M:%S"`

module purge
# module load slurm/20.02.7
# module load tensorflow2-py37-cuda10.2-gcc8/2.5.0  
# module load python-mpi4py-3.0.3
# module list

source venv/bin/activate
# conda activate patch
which python3
python3 --version

# srun python3 -m src.run
# python3 -m source.run
python3 -m codellama

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`