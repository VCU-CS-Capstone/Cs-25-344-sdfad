#!/bin/bash
#SBATCH --job-name=avx_debug
#SBATCH --output=slurm_logs/avx_debug.log
#SBATCH --error=slurm_logs/avx_debug.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo_env

python avx_debug_test.py
