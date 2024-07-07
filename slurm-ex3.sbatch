#!/bin/bash
#SBATCH -p dgx2q
#SBATCH  --job-name=T5
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time 01-00:00:00    # time (D-HH:MM:SS)
#SBATCH -o ./output/%j.out # STDOUT
# SBATCH -e ./output/%j.err # STDERR

ulimit -s 10240
echo "Job started at:" `date +"%Y-%m-%d %H:%M:%S"`

module purge
module load slurm/20.02.7
# module load tensorflow2-py37-cuda10.2-gcc8/2.5.0  
# module load python-mpi4py-3.0.3
# module list

conda activate patch
which python3
python3 --version

# srun python3 -m src.run
python3 -m source.run

echo "Job ended at:" `date +"%Y-%m-%d %H:%M:%S"`