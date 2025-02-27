import torch
import sys  # Used for exiting if GPU isn't available

# ---------------------
# 🔹 DEBUG STEP 1: CHECK FOR GPU
# ---------------------
if not torch.cuda.is_available():
    print("❌ No GPU detected. Exiting program.")
    sys.exit(1)

device = torch.device("cuda")
print(f"🚀 Running on: {device}")
print(f"PyTorch Version: {torch.__version__}")

# Test tensor computation on GPU
try:
    test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
    print(f"✅ GPU test successful: {test_tensor}")
except Exception as e:
    print(f"❌ GPU test failed: {e}")
    sys.exit(1)

# ---------------------
# 🔹 IMPORT OTHER LIBRARIES
# ---------------------
import torchvision.transforms as transforms
from ultralytics import YOLO
# import open3d as o3d  # ❌ DISABLED Open3D to avoid "Illegal Instruction"
import scipy.io
import numpy as np
import pandas as pd
import glob
import os
import argparse
from PIL import Image

print("⚠️ Open3D has been disabled to prevent potential 'Illegal Instruction' errors.")

# ---------------------
# 🔹 PARSE ARGUMENTS
# ---------------------
parser = argparse.ArgumentParser(description="Annotate images and LiDAR data in multiple folders.")
parser.add_argument("dataset_dir", type=str, help="Path to the parent dataset folder containing multiple subfolders.")
args = parser.parse_args()

PARENT_DATASET_DIR = args.dataset_dir  # Parent dataset directory

# ---------------------
# 🔹 FUNCTION TO MATCH IMAGE AND LIDAR TIMESTAMPS
# ---------------------
def match_timestamps(img_timestamps, lidar_timestamps):
    """
    Matches each image timestamp to the closest LiDAR timestamp.
    Returns a list of (image_index, lidar_index) pairs.
    """
    matched_indices = []
    for img_idx, img_ts in enumerate(img_timestamps):
        closest_lidar_idx = np.argmin(np.abs(lidar_timestamps - img_ts))
        matched_indices.append((img_idx, closest_lidar_idx))
    return matched_indices

# ---------------------
# 🔹 LOAD PRE-TRAINED YOLOv8 FOR IMAGE DETECTION
# ---------------------
model = YOLO("yolov8n.pt").to(device)  # Move model to GPU

# ---------------------
# 🔹 PROCESS EACH SUBFOLDER IN THE PARENT DIRECTORY
# ---------------------
subfolders = [os.path.join(PARENT_DATASET_DIR, d) for d in os.listdir(PARENT_DATASET_DIR) if os.path.isdir(os.path.join(PARENT_DATASET_DIR, d))]

for DATASET_DIR in subfolders:
    print(f"📂 Processing folder: {DATASET_DIR}")

    # Paths
    IMG_DIR = os.path.join(DATASET_DIR, "img")
    LIDAR_DIR = os.path.join(DATASET_DIR, "velo")
    ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations")
    os.makedirs(ANNOTATION_DIR, exist_ok=True)  # Ensure the annotations folder exists

    # Load timestamps
    try:
        img_timestamps = pd.read_csv(os.path.join(DATASET_DIR, "imgtimestamps.csv")).values[:, 1]
        lidar_timestamps = pd.read_csv(os.path.join(DATASET_DIR, "velotimestamps.csv")).values[:, 1]
    except Exception as e:
        print(f"❌ Error loading timestamps in {DATASET_DIR}: {e}")
        continue  # Skip this folder if there's an issue

    # Get matched indices
    matched_pairs = match_timestamps(img_timestamps, lidar_timestamps)

    # Get file lists
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    lidar_files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.pcd")))

    for img_idx, lidar_idx in matched_pairs:
        if img_idx >= len(img_files) or lidar_idx >= len(lidar_files):
            continue  # Skip if index is out of range

        img_path = img_files[img_idx]
        lidar_path = lidar_files[lidar_idx]

        # Run YOLO on the image (send image to GPU)
        results = model(img_path)

        # Collect detections
        detections = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xywh, r.boxes.conf, r.boxes.cls):
                x, y, w, h = box.cpu().numpy()  # Ensure results are sent back to CPU
                confidence = conf.cpu().numpy()
                label = int(cls.cpu().numpy())

                # If the detected object is a pedestrian (YOLO class 0)
                if label == 0:
                    detections.append([x, y, 0, w, h, 0, confidence])  # 0 = Pedestrian type

        # ---------------------
        # 🔹 SKIP LiDAR PROCESSING (Open3D Disabled)
        # ---------------------
        print("⚠️ Skipping LiDAR processing because Open3D is disabled.")

        # ---------------------
        # 🔹 SAVE ANNOTATIONS AS .MAT FILES
        # ---------------------
        annotation_file = os.path.join(ANNOTATION_DIR, "detect" + os.path.basename(img_path).replace(".png", ".mat"))
        scipy.io.savemat(annotation_file, {"detection": np.array(detections)})

        print(f"✅ Saved annotation: {annotation_file}")

print("🎉 Annotation generation complete!")
