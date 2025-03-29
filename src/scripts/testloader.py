import torch
from dataset import WaymoParquetDataset

# Clearly specify your base directory and scene IDs
base_dir = './waymo_data/training'
scene_ids = ['14073491244121877213_4066_056_4086_056', '14106113060128637865_1200_000_1220_000']  # Replace explicitly with actual scene IDs you have

# Initialize dataset explicitly
dataset = WaymoParquetDataset(scene_ids=scene_ids, base_dir=base_dir)

# Verify dataset length explicitly
print("Dataset size:", len(dataset))

# Load a single sample explicitly
sample = dataset[0]

# Print clearly to verify loading correctness
print("RGB Image shape:", sample['rgb_image'].shape)
print("LiDAR Range Image shape:", sample['lidar_range_image'].shape)
print("Bounding boxes shape:", sample['boxes'].shape)
print("Confidence scores:", sample['confidences'])
print("Calibration data (camera) columns:", sample['camera_calibration'].columns)
print("Calibration data (LiDAR) columns:", sample['lidar_calibration'].columns)
