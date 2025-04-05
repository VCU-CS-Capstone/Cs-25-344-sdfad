#!/bin/bash
#SBATCH --job-name=explore_parquet
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

# Move to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Define project base and output directory
BASE_DIR="/home/cs-25-344/Cs-25-344-sdfad"
SCRIPT="$BASE_DIR/src/scripts/explore_parquet.py"
OUTPUT_DIR="$BASE_DIR/src/outputs"

# Optional: Activate conda or load modules
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# Run the script
echo "Running $SCRIPT"
python "$SCRIPT"

# === Fix file permissions and group after the script completes ===
echo "Fixing output permissions..."
chgrp -R 'egr cs-25-344' "$OUTPUT_DIR"
chmod -R g+rw "$OUTPUT_DIR"
chmod -R o-rwx "$OUTPUT_DIR"

echo "Done at: $(date)"
