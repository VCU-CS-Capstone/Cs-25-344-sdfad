#!/bin/bash
#SBATCH --job-name=Download_Waymo
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

# Ensure we run from the submission directory
cd $SLURM_SUBMIT_DIR

echo "===== Starting Full Waymo Download: All Folders ====="
echo "Time: $(date)"
echo "====================================================="

FOLDERS=("lidar" "camera_image" "lidar_box" "lidar_calibration" "camera_calibration" "camera_box")
BASE_GCS="gs://waymo_open_dataset_v_2_0_1/training"
BASE_LOCAL="/home/cs-25-344/waymo_data/training"

mkdir -p "$BASE_LOCAL"

# Optional: ensure group ownership and access
newgrp 'egr cs-25-344'
umask 007

# Step 1: Loop through each folder and download all contents
for folder in "${FOLDERS[@]}"; do
    echo ">>> Downloading all files for $folder..."
    mkdir -p "$BASE_LOCAL/$folder"
    
    gsutil ls "$BASE_GCS/$folder/*.parquet" > "${folder}_list.txt"
    gsutil -m cp -I "$BASE_LOCAL/$folder/" < "${folder}_list.txt"
done

# Step 2: Clean up any temporary .gstmp files
echo ">>> Cleaning up temporary .gstmp files..."
find "$BASE_LOCAL" -type f -name "*.gstmp" -exec rm -v {} \;

echo "===== Full Download Complete ====="
echo "Time: $(date)"
