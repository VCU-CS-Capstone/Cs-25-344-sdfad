#!/bin/bash
#SBATCH --job-name=waymo_dl_camera_box
#SBATCH --output=waymo_dl.log
#SBATCH --error=waymo_dl.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0

echo "===== Resuming Waymo Download: Camera Box & Any Missing ====="
echo "Time: $(date)"
echo "=============================================================="

mkdir -p $HOME/waymo_data/training

# Step 1: Get the first 399 LiDAR filenames if not already done
if [ ! -f lidar_half.txt ]; then
    echo "Generating lidar_half.txt from gsutil listing..."
    gsutil ls gs://waymo_open_dataset_v_2_0_1/training/lidar/*.parquet | head -n 399 > lidar_half.txt
else
    echo "Using existing lidar_half.txt"
fi

# Step 2: Updated list of folders including only camera_box
#FOLDERS=("lidar" "camera_image" "lidar_box" "lidar_calibration" "camera_calibration" "camera_box")
FOLDERS=("camera_box")

# Step 3: Download only missing files
for folder in "${FOLDERS[@]}"; do
    echo ">>> Downloading missing files for $folder..."
    mkdir -p "$HOME/waymo_data/training/$folder"

    sed "s|/lidar/|/$folder/|" lidar_half.txt | \
    while read -r cloud_path; do
        filename=$(basename "$cloud_path")
        local_path="$HOME/waymo_data/training/$folder/$filename"

        if [ -f "$local_path" ]; then
            echo "Skipping (exists): $filename"
        else
            echo "Downloading: $cloud_path"
            gsutil cp "$cloud_path" "$local_path"
        fi
    done
done

# Step 4: Clean up any temporary partials
echo ">>> Cleaning up temporary .gstmp files..."
find $HOME/waymo_data/training -type f -name "*.gstmp" -exec rm -v {} \;

echo "===== Download Complete (Selective) ====="
echo "Time: $(date)"
