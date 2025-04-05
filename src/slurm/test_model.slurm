#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

# Move to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Define base project dir and target script
BASE_DIR="/home/cs-25-344/Cs-25-344-sdfad"
SCRIPT="$BASE_DIR/src/model_and_training/model.py"
OUTPUT_DIR="$BASE_DIR/src/outputs"

# Activate conda environment
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# Run the model test script
echo "🔁 Running model.py test block..."
python "$SCRIPT"

# Fix output file permissions
echo "🔧 Fixing output permissions..."
chgrp -R 'egr cs-25-344' "$OUTPUT_DIR"
chmod -R g+rw "$OUTPUT_DIR"
chmod -R o-rwx "$OUTPUT_DIR"

echo "✅ Done at: $(date)"
