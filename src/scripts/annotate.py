import torch
import sys  # Used for exiting if GPU isn't available
import torchvision.transforms as transforms
from ultralytics import YOLO
import scipy.io
import numpy as np
import pandas as pd
import glob
import os
import argparse
import struct
import zlib

# ---------------------
# 🔹 CHECK FOR GPU
# ---------------------
if not torch.cuda.is_available():
    print("❌ No GPU detected. Exiting program.")
    sys.exit(1)

device = torch.device("cuda")
print(f"🚀 Running on: {device}")
print(f"PyTorch Version: {torch.__version__}")

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
    matched_indices = []
    for img_idx, img_ts in enumerate(img_timestamps):
        closest_lidar_idx = np.argmin(np.abs(lidar_timestamps - img_ts))
        matched_indices.append((img_idx, closest_lidar_idx))
    return matched_indices

# ---------------------
# 🔹 FUNCTION TO READ COMPRESSED BINARY PCD FILE
# ---------------------
def load_pcd_binary_compressed(file_path):
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            if line.startswith("DATA"):
                data_format = line.split()[-1].lower()
                break
            header.append(line)

        assert data_format == "binary_compressed", "This function only supports compressed binary PCD files."

        metadata = {line.split()[0]: line.split()[1:] for line in header}
        num_points = int(metadata["POINTS"][0])
        fields = metadata["FIELDS"]
        field_types = metadata["TYPE"]
        field_sizes = list(map(int, metadata["SIZE"]))

        bytes_per_point = sum(field_sizes)

        compressed_size = struct.unpack("I", f.read(4))[0]
        uncompressed_size = struct.unpack("I", f.read(4))[0]

        compressed_data = f.read(compressed_size)
        uncompressed_data = zlib.decompress(compressed_data)

        dtype_list = [(fields[i], field_types[i] + str(field_sizes[i])) for i in range(len(fields))]
        point_cloud_array = np.frombuffer(uncompressed_data, dtype=np.dtype(dtype_list), count=num_points)

        tensor_data = torch.tensor([list(point) for point in point_cloud_array], dtype=torch.float32)
        return tensor_data

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

    IMG_DIR = os.path.join(DATASET_DIR, "img")
    LIDAR_DIR = os.path.join(DATASET_DIR, "velo")
    ANNOTATION_DIR = os.path.join(DATASET_DIR, "annotations")
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    try:
        img_timestamps = pd.read_csv(os.path.join(DATASET_DIR, "imgtimestamps.csv")).values[:, 1]
        lidar_timestamps = pd.read_csv(os.path.join(DATASET_DIR, "velotimestamps.csv")).values[:, 1]
    except Exception as e:
        print(f"❌ Error loading timestamps in {DATASET_DIR}: {e}")
        continue

    matched_pairs = match_timestamps(img_timestamps, lidar_timestamps)

    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    lidar_files = sorted(glob.glob(os.path.join(LIDAR_DIR, "*.pcd")))

    for img_idx, lidar_idx in matched_pairs:
        if img_idx >= len(img_files) or lidar_idx >= len(lidar_files):
            continue

        img_path = img_files[img_idx]
        lidar_path = lidar_files[lidar_idx]

        # ---------------------
        # 🔹 IMAGE PROCESSING WITH YOLO
        # ---------------------
        results = model(img_path)

        detections = []
        for r in results:
            for box, conf, cls in zip(r.boxes.xywh, r.boxes.conf, r.boxes.cls):
                x, y, w, h = box.cpu().numpy()
                confidence = conf.cpu().numpy()
                label = int(cls.cpu().numpy())

                if label == 0:  # Pedestrian class
                    detections.append([x, y, 0, w, h, 0, confidence])

        # ---------------------
        # 🔹 LiDAR PROCESSING WITHOUT OPEN3D
        # ---------------------
        try:
            lidar_tensor = load_pcd_binary_compressed(lidar_path)
            print(f"✅ Loaded LiDAR tensor: {lidar_path}, Shape: {lidar_tensor.shape}")
        except Exception as e:
            print(f"❌ Error loading LiDAR file {lidar_path}: {e}")
            continue

        # ---------------------
        # 🔹 SAVE ANNOTATIONS AS .MAT FILES
        # ---------------------
        annotation_file = os.path.join(ANNOTATION_DIR, "detect" + os.path.basename(img_path).replace(".png", ".mat"))
        scipy.io.savemat(annotation_file, {"detection": np.array(detections)})

        print(f"✅ Saved annotation: {annotation_file}")

print("🎉 Annotation generation complete!")
