#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

cd $SLURM_SUBMIT_DIR

BASE_DIR="/home/cs-25-344/Cs-25-344-sdfad"
SCRIPT="$BASE_DIR/src/model_and_training/train.py"

SanityTest="--epochs 1 --batch_size 4 --modality fusion --debug_interval 1 --dataset_limit 10"  # sanity test 5min
Quick="--epochs 1 --batch_size 4 --modality fusion --debug_interval 2 --dataset_limit 40"  # quick check 15 min
TrainTrial="--epochs 3 --batch_size 8 --modality fusion --debug_interval 5 --dataset_limit 100" # training trial 1hr
TrainTrialBig="--epochs 5 --batch_size 8 --modality fusion --debug_interval 10 --dataset_limit 200" # bigger training trial 3hr
HalfSet="--epochs 10 --batch_size 8 --modality fusion --debug_interval 20 --dataset_limit 400" # half set 8hr
full="--epochs 20 --batch_size 8 --modality fusion --debug_interval 50" # full train ~2 days

cameraFull="--epochs 20 --batch_size 8 --modality camera --debug_interval 50"  # full train ~2 days (but will run faster)


# === Choose config ===
CONFIG_NAME="cameraFull"
ARGS="$cameraFull"

OUTPUT_DIR="$BASE_DIR/src/outputs"

source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

echo "🧪 Running with: $CONFIG_NAME"
echo "⚙️  Arguments: $ARGS"

START=$(date +%s)
echo "🚀 Starting training run on $(hostname) @ $(date)"

python "$SCRIPT" $ARGS

echo "✅ Training complete."

# === FIX OUTPUT PERMISSIONS ===
echo "🔧 Fixing output permissions..."
chgrp -R 'egr cs-25-344' "$OUTPUT_DIR"
chmod -R g+rw "$OUTPUT_DIR"
chmod -R o-rwx "$OUTPUT_DIR"

END=$(date +%s)
DURATION=$((END - START))
DAYS=$((DURATION / 86400))
HOURS=$(( (DURATION % 86400) / 3600 ))
MINS=$(( (DURATION % 3600) / 60 ))
SECS=$((DURATION % 60))

echo "🏁 Finished at: $(date)"
echo "⌚ Elapsed time: ${DAYS}d ${HOURS}h ${MINS}m ${SECS}s"
