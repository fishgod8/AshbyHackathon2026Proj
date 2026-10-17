#!/bin/bash
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=prepare_bach.out
#SBATCH --error=prepare_bach.err

module load pytorch-conda/2.8
python prepare_bach.py
