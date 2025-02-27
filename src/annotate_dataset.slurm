#!/bin/bash
#SBATCH --job-name=annotate_dataset
#SBATCH --output=annotate_output.log
#SBATCH --error=annotate_error.log
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu      # Ensure it runs on a GPU node
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH --export=ALL

echo "🚀 Starting annotation at $(date) on $(hostname)"
echo "🔍 Checking SLURM-assigned GPU resources..."
scontrol show job $SLURM_JOB_ID | grep Gres

echo "🔍 Checking available GPUs on this node..."
nvidia-smi

echo "🔍 Python version being used..."
which python
python --version

echo "🔍 Testing YOLO before running annotation..."
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('✅ YOLO loaded successfully!')"

echo "🔍 Checking if PyTorch detects the GPU..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Using Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"


echo "🔍 Running the annotation script..."
python ~/annotate.py ~/dataset

echo "✅ job completed at $(date)"
