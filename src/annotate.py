import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import open3d as o3d
import scipy.io
import numpy as np
import pandas as pd
import glob
import os
import argparse
from PIL import Image

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
model = YOLO("yolov8n.pt")  # YOLOv8 nano model for pedestrian detection

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

        # Run YOLO on the image
        results = model(img_path)

        # Collect detections
        detections = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xywh, r.boxes.conf, r.boxes.cls):
                x, y, w, h = box.numpy()
                confidence = conf.numpy()
                label = int(cls.numpy())

                # If the detected object is a pedestrian (YOLO class 0)
                if label == 0:
                    detections.append([x, y, 0, w, h, 0, confidence])  # 0 = Pedestrian type

        # ---------------------
        # 🔹 PROJECT IMAGE DETECTIONS TO LIDAR
        # ---------------------
        pcd = o3d.io.read_point_cloud(lidar_path)  # Load LiDAR .pcd file
        points = np.asarray(pcd.points)  # Get (x, y, z) coordinates

        # Convert LiDAR points to homogeneous coordinates
        lidar_points_h = np.hstack((points, np.ones((points.shape[0], 1))))

        # Transformation Matrix (Replace with actual LiDAR-to-Camera transformation)
        T_LIDAR_TO_CAMERA = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])  # 🚨 Replace this with your actual dataset calibration matrix!

        # Transform LiDAR points into camera space
        projected_points = lidar_points_h @ T_LIDAR_TO_CAMERA.T

        # Keep only points within detected pedestrian bounding boxes
        lidar_pedestrian_points = []
        for detection in detections:
            x, y, _, w, h, _, conf = detection  # Get image bounding box

            # Filter LiDAR points within the pedestrian bounding box (Approximate method)
            inside_box = (projected_points[:, 0] >= x - w / 2) & (projected_points[:, 0] <= x + w / 2) & \
                         (projected_points[:, 1] >= y - h / 2) & (projected_points[:, 1] <= y + h / 2)

            pedestrian_points = points[inside_box]  # Keep only pedestrian LiDAR points
            if pedestrian_points.shape[0] > 0:
                lidar_pedestrian_points.append(pedestrian_points)

        # ---------------------
        # 🔹 SAVE ANNOTATIONS AS .MAT FILES
        # ---------------------
        annotation_file = os.path.join(ANNOTATION_DIR, "detect" + os.path.basename(img_path).replace(".png", ".mat"))
        scipy.io.savemat(annotation_file, {"detection": np.array(detections)})

        print(f"✅ Saved annotation: {annotation_file}")

print("🎉 Annotation generation complete!")
