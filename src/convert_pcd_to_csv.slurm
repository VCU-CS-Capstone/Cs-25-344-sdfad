#!/bin/bash
#SBATCH --job-name=convert_pcd
#SBATCH --output=convert_pcd_output.log
#SBATCH --error=convert_pcd_error.log
#SBATCH --time=10:00       # Short test job (10 minutes)
#SBATCH --mem=4G
#SBATCH --partition=gpu    # Running on GPU partition
#SBATCH --gres=gpu:1       # Request 1 GPU (can be changed if not needed)
#SBATCH --export=ALL

echo "🚀 Starting PCD to CSV conversion at $(date) on $(hostname)"

# Check Python version
echo "🔍 Python version being used:"
which python
python --version

# Ensure required packages are installed
echo "🔍 Checking if numpy & pandas are installed..."
python -c "import numpy, pandas; print('✅ numpy & pandas are installed!')" || { echo "❌ ERROR: Missing dependencies! Install with: pip install --user numpy pandas"; exit 1; }

# Set file paths
PCD_FILE=~/dataset/_2019-02-09-13-04-51/velo/velo0000.pcd
CSV_FILE=~/dataset/velo0000.csv

# Run conversion script
echo "🔍 Running conversion script on $PCD_FILE..."
python ~/convert_pcd_to_csv.py "$PCD_FILE" "$CSV_FILE"

echo "✅ PCD to CSV conversion completed at $(date)"
