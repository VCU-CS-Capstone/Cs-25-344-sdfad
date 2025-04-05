#!/bin/bash
#SBATCH --job-name=train_model_test
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=01:00:00                # Shorter time for test run
#SBATCH --cpus-per-task=1              # For DataLoader
#SBATCH --mem=32G                      # Smaller mem for lighter run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                   # 1x A100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

# Move to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# === PATH CONFIG ===
BASE_DIR="/home/cs-25-344/Cs-25-344-sdfad"
SCRIPT="$BASE_DIR/src/model_and_training/train.py"
ARGS="--epochs 1 --batch_size 4 --modality fusion --debug_interval 10 --dataset_limit 40"
OUTPUT_DIR="$BASE_DIR/src/outputs"

# === ENVIRONMENT ===
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# === RUN TEST TRAINING ===
echo "🚀 Starting TEST training run on $(hostname)"
python "$SCRIPT" $ARGS
echo "✅ Test training complete."

# === FIX OUTPUT PERMISSIONS ===
echo "🔧 Fixing output permissions..."
chgrp -R 'egr cs-25-344' "$OUTPUT_DIR"
chmod -R g+rw "$OUTPUT_DIR"
chmod -R o-rwx "$OUTPUT_DIR"

echo "🏁 Finished at: $(date)"
