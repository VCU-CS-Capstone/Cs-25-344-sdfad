#!/bin/bash
#SBATCH --job-name=waymo_testloader
#SBATCH --output=testloader.out
#SBATCH --error=testloader.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

echo "===== Starting testloader.py ====="
echo "Time: $(date)"
echo "=================================="

# Activate your environment (adjust if named differently)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo_env

# Run your test script
python testloader.py

echo "===== Finished testloader.py ====="
echo "Time: $(date)"
