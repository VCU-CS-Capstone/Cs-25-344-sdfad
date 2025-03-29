#!/bin/bash
#SBATCH --job-name=parquet_test
#SBATCH --output=slurm_logs/test_parquet_minimal.log
#SBATCH --error=slurm_logs/test_parquet_minimal.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo_env

python ./scripts/test_parquet_minimal.py
