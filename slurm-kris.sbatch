#!/bin/bash
#SBATCH --job-name=PatchT5          # Job name
#SBATCH --partition=LocalQ             # Partition to submit to
#SBATCH --output=output/out.%j         # Output file name (%j expands to jobID)
# SBATCH --error=output/%j-err.txt               # Error file name (%j expands to jobID)
#SBATCH --ntasks=4                     # Number of tasks (MPI processes)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=4            # Number of tasks per node
#SBATCH --time=03-00:10:00                # Walltime (hh:mm:ss)

# activate the conda environment
source /cluster/storage/anaconda3/bin/activate 
conda activate fixme


which python3
whereis python3

# Run your program here
python3 -m generator.run
# python3 test.py