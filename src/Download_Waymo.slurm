#!/bin/bash
#SBATCH --job-name=waymo_dl
#SBATCH --output=waymo_dl.log
#SBATCH --error=waymo_dl.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

echo "===== Starting Half-Waymo Download ====="
echo "Time: $(date)"
echo "========================================"

mkdir -p $HOME/waymo_data/training

# Step 1: Get the first 399 filenames from lidar
echo "Listing first 399 LiDAR files..."
gsutil ls gs://waymo_open_dataset_v_2_0_1/training/lidar/*.parquet | head -n 399 > lidar_half.txt

FOLDERS=("lidar" "camera_image" "lidar_box" "lidar_calibration" "camera_calibration")

# Step 2: Use lidar_half.txt to match across all folders
for folder in "${FOLDERS[@]}"; do
    echo ">>> Downloading aligned files for $folder..."
    mkdir -p $HOME/waymo_data/training/$folder
    sed "s|/lidar/|/$folder/|" lidar_half.txt | \
    xargs -I {} sh -c 'echo "Now downloading: {}"; gsutil cp {} $HOME/waymo_data/training/'"$folder"'/'
done

echo ">>> Cleaning up temporary .gstmp files..."
find $HOME/waymo_data/training -type f -name "*.gstmp" -exec rm -v {} \;

echo "===== Half Dataset Download Complete ====="
echo "Time: $(date)"
