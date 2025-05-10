#!/bin/bash
#SBATCH --job-name=mass_infer_compare
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

# === CONFIG ===
LIMIT=15  # number of scenes to visualize
OUTDIR="/home/cs-25-344/Cs-25-344-sdfad/src/outputs/comparison_slideshow"
SCRIPT="/home/cs-25-344/Cs-25-344-sdfad/src/model_and_training/mass_inference.py"
ARGS="--limit $LIMIT --outdir $OUTDIR"

# === ENV SETUP ===
cd $SLURM_SUBMIT_DIR
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# === RUN ===
echo "🔍 Running mass comparison inference (limit $LIMIT)"
python "$SCRIPT" $ARGS
echo "✅ Done. Images saved to $OUTDIR"
