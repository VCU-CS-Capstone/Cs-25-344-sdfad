#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --output=slurm_logs/install_deps.log
#SBATCH --error=slurm_logs/install_deps.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

echo "===== Starting Dependency Installation for Tesla (AVX-safe) ====="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate waymo_env

# Clean up potential conflicts from user-level pip packages
echo "Cleaning ~/.local/lib/python3.9/site-packages/..."
rm -rf ~/.local/lib/python3.9/site-packages/*

# Install CPU-only PyTorch stack (safe versions for older CPUs like Tesla's)
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 cpuonly -c pytorch --yes

# Core data handling + image support (safe versions)
conda install numpy=1.21.6 pandas=1.3.5 pyarrow=8.0.0 opencv=4.5.5 -c conda-forge --yes

# Optional: common utilities
conda install matplotlib=3.5.3 scikit-learn=1.0.2 -c conda-forge --yes

# EfficientNet for model (pip only, AVX-safe older version)
pip install efficientnet_pytorch==0.7.1

echo "===== Finished Installing Dependencies ====="
