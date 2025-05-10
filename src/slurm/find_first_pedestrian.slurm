#!/bin/bash
#SBATCH --job-name=find_pedestrian
#SBATCH --output=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.out
#SBATCH --error=/home/cs-25-344/Cs-25-344-sdfad/src/outputs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonyde@vcu.edu

cd $SLURM_SUBMIT_DIR

# === ENV SETUP ===
source /home/cs-25-344/miniconda3/etc/profile.d/conda.sh
conda activate /home/cs-25-344/conda-envs/waymo-env

# === RUN SCRIPT ===
echo "🔍 Starting pedestrian sample scan @ $(date)"
python <<EOF
import os
import pyarrow.parquet as pq

root_dir = "/home/cs-25-344/waymo_data/training"
camera_box_dir = os.path.join(root_dir, "camera_box")

# fallback pedestrian type enum
try:
    from waymo_open_dataset.label_pb2 import Label
    TYPE_PEDESTRIAN = Label.Type.Value('TYPE_PEDESTRIAN')
except:
    TYPE_PEDESTRIAN = 1  # fallback enum ID used in most datasets

for fname in sorted(os.listdir(camera_box_dir)):
    if not fname.endswith(".parquet"):
        continue
    path = os.path.join(camera_box_dir, fname)
    base_name = fname[:-8]
    try:
        table = pq.read_table(path).to_pandas()
    except Exception as e:
        print(f"❌ Failed to read {fname}: {e}")
        continue

    if TYPE_PEDESTRIAN not in table['[CameraBoxComponent].type'].values:
        continue

    print(f"✅ Found pedestrian in: {base_name}")
    break
else:
    print("❌ No pedestrian found.")
EOF

echo "✅ Done @ $(date)"
