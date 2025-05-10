#!/bin/bash
#SBATCH --job-name=infer_visual
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

# === CONFIG ===
BASE_DIR="/home/cs-25-344/Cs-25-344-sdfad"
SCRIPT="$BASE_DIR/src/scripts/find_and_infer_many.py"
ARGS="--max 20"

# === SETUP ENV ===
cd $SLURM_SUBMIT_DIR
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# === RUN ===
echo "🔍 Running batch inference on up to 20 pedestrian samples..."
python "$SCRIPT" $ARGS

echo "✅ Done at $(date)"
