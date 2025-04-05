import os
import io
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pyarrow.parquet as pq
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2


class WaymoFusionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_lidar_points=100000, limit=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.max_lidar_points = max_lidar_points

        # Collect all synchronized filenames (excluding extension)
        self.filenames = self._get_synchronized_filenames()
        if limit:
            self.filenames = self.filenames[:limit]

    def _get_synchronized_filenames(self):
        cam_files = set(f[:-8] for f in os.listdir(os.path.join(self.root_dir, 'camera_image')) if f.endswith('.parquet'))
        lidar_files = set(f[:-8] for f in os.listdir(os.path.join(self.root_dir, 'lidar')) if f.endswith('.parquet'))
        common = sorted(cam_files & lidar_files)
        return common

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        base_name = self.filenames[idx]

        rgb = self._load_rgb_image(base_name)
        lidar = self._load_lidar_points(base_name)
        targets = self._load_pedestrian_labels(base_name)

        return rgb, lidar, targets

    def _load_rgb_image(self, base_name):
        cam_path = os.path.join(self.root_dir, 'camera_image', base_name + '.parquet')
        table = pq.read_table(cam_path).to_pandas()
        image_bytes = table['[CameraImageComponent].image'].iloc[0]
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(img)

    def _load_lidar_points(self, base_name):
        lidar_path = os.path.join(self.root_dir, 'lidar', base_name + '.parquet')
        table = pq.read_table(lidar_path).to_pandas()

        # Directly fetch the raw LiDAR data (now stored as a NumPy array)
        raw_data = table['[LiDARComponent].range_image_return1.values'].iloc[0]
        points = np.array(raw_data).reshape(-1, 4)  # Reshaped to (N, 4) format (x, y, z, intensity)

        # Handle cases where there are more than 4 columns (just drop extras)
        if points.shape[1] > 4:
            points = points[:, :4]

        # Padding if there are fewer than max_lidar_points
        if points.shape[0] > self.max_lidar_points:
            points = points[:self.max_lidar_points]
        else:
            padding = np.zeros((self.max_lidar_points - points.shape[0], 4))
            points = np.vstack([points, padding])

        return torch.tensor(points, dtype=torch.float32)

    def _load_pedestrian_labels(self, base_name):
        box_path = os.path.join(self.root_dir, 'camera_box', base_name + '.parquet')
        table = pq.read_table(box_path).to_pandas()

        ped_boxes = table[table['[CameraBoxComponent].type'] == label_pb2.Label.Type.TYPE_PEDESTRIAN]
        boxes = []
        for _, row in ped_boxes.iterrows():
            x = row['[CameraBoxComponent].box.center.x']
            y = row['[CameraBoxComponent].box.center.y']
            w = row['[CameraBoxComponent].box.size.x']
            h = row['[CameraBoxComponent].box.size.y']
            boxes.append([x, y, w, h])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.ones(len(boxes), dtype=torch.long)
        }
        return target


if __name__ == "__main__":
    root_path = "/home/cs-25-344/waymo_data/training/"
    dataset = WaymoFusionDataset(root_path)
    print(f"Dataset length: {len(dataset)}")

    sample_idx = 0
    rgb, lidar, target = dataset[sample_idx]

    print("Sample RGB image shape:", rgb.shape)
    print("Sample LiDAR tensor shape:", lidar.shape)
    print("Target dictionary:", target)
    print("Number of pedestrian boxes:", len(target['boxes']))
