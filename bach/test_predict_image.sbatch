#!/bin/bash
#SBATCH --job-name=bach-predict
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=bach-predict-%j.out
#SBATCH --error=bach-predict-%j.err

set -euo pipefail

module load pytorch-conda/2.8
unset PYTHONNOUSERSITE

python bach/predict_image.py \
  --image /work/nvme/bfqi/$USER/Photos/Benign/b028.tif
