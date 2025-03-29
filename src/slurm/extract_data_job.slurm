#!/bin/bash
#SBATCH --job-name=unzip_dataset
#SBATCH --output=unzip_output.log
#SBATCH --error=unzip_error.log
#SBATCH --time=1:00:00    # Set max time (adjust if needed)
#SBATCH --mem=4G          # Adjust memory if needed
#SBATCH --cpus-per-task=1 # Use multiple CPU cores
#SBATCH --partition=gpu  # Replace with the correct partition name

echo "Starting extraction on $(hostname) at $(date)"
mkdir -p ~/dataset

for zipfile in ~/dataset_download/extracted_data/*.zip; do
    echo "Extracting $zipfile..."
    if unzip -o "$zipfile" -d ~/dataset; then
        echo "Successfully extracted $zipfile, deleting..."
        rm "$zipfile"
    else
        echo "❌ Failed to extract $zipfile, skipping deletion."
    fi
done

echo "Extraction completed at $(date)"
