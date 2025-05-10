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
SAMPLE_ID="12179768245749640056_5561_070_5581_070"
#SAMPLE_ID="10017090168044687777_6380_000_6400_000"
#SAMPLE_ID="10023947602400723454_1120_000_1140_000"

BASE_DIR="/home/cs-25-344/Cs-25-344-sdfad"

#SCRIPT="$BASE_DIR/src/model_and_training/inference_script.py"
SCRIPT="$BASE_DIR/src/model_and_training/mass_inference.py"

OUT_IMG="$BASE_DIR/src/outputs/infer_${SAMPLE_ID}.png"
ARGS="--single-sample $SAMPLE_ID --out $OUT_IMG"
#ARGS="--single-sample $SAMPLE_ID --out $OUT_IMG --no-infer --no-boxes"

# === SETUP ENV ===
cd $SLURM_SUBMIT_DIR
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# === RUN ===
echo \"🔍 Running inference on $SAMPLE_ID\"
python "$SCRIPT" $ARGS

echo \"✅ Done: image saved to $OUT_IMG\"
