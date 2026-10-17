#!/bin/bash
#SBATCH --job-name=bach-train
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=train_bach.out
#SBATCH --error=train_bach.err

module load pytorch-conda/2.8
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

python train_bach.py
