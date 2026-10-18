#!/bin/bash
#SBATCH --job-name=similarity_search
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:10:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Load the required Python environment or module
module reset
module load pytorch-conda/2.8
unset PYTHONNOUSERSITE
# Run your Python script
srun python3 similarity_search.py