#!/bin/bash
#SBATCH --job-name=ds_only
#SBATCH --output=slurm_logs/test_dataset_only.log
#SBATCH --error=slurm_logs/test_dataset_only.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo_env

python ./scripts/dataset.py