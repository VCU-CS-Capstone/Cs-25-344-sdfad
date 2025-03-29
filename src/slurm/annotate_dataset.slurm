#!/bin/bash
#SBATCH --job-name=annotate_dataset
#SBATCH --output=annotate_output.log
#SBATCH --error=annotate_error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

# Load CUDA (ensure correct version, adjust if needed)
module load cuda/12.1  # This matches your installed PyTorch CUDA version

# Activate Conda environment
source ~/miniconda3/bin/activate pytorch310  # Ensure this matches your working env

# Check PyTorch inside the job (debugging)
python -c "import torch; print('✅ PyTorch Loaded:', torch.__version__)"

# Run the Python script
python ~/annotate.py ~/dataset
