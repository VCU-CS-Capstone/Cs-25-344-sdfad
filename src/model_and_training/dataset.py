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

try:
    CAMERA_FRONT = label_pb2.CameraName.Value("FRONT")
    TYPE_PEDESTRIAN = label_pb2.Label.Type.Value("TYPE_PEDESTRIAN")
except Exception:
    CAMERA_FRONT = 1
    TYPE_PEDESTRIAN = 2


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

        rgb, orig_size = self._load_rgb_image(base_name)
        lidar = self._load_lidar_points(base_name)
        targets = self._load_pedestrian_labels(base_name, orig_size)

        return rgb, lidar, targets

    def _load_rgb_image(self, base_name):
        cam_path = os.path.join(self.root_dir, 'camera_image', base_name + '.parquet')
        table = pq.read_table(cam_path).to_pandas()
        front_row = table[table['key.camera_name'] == CAMERA_FRONT].iloc[0]
        image_bytes = front_row['[CameraImageComponent].image']
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = img.size
        return self.transform(img), orig_size

    def _load_lidar_points(self, base_name):
        lidar_path = os.path.join(self.root_dir, 'lidar', base_name + '.parquet')
        table = pq.read_table(lidar_path).to_pandas()

        raw_data = table['[LiDARComponent].range_image_return1.values'].iloc[0]
        points = np.array(raw_data).reshape(-1, 4)

        if points.shape[1] > 4:
            points = points[:, :4]

        if points.shape[0] > self.max_lidar_points:
            points = points[:self.max_lidar_points]
        else:
            padding = np.zeros((self.max_lidar_points - points.shape[0], 4))
            points = np.vstack([points, padding])

        return torch.tensor(points, dtype=torch.float32)

    def _load_pedestrian_labels(self, base_name, orig_size):
        box_path = os.path.join(self.root_dir, 'camera_box', base_name + '.parquet')
        table = pq.read_table(box_path).to_pandas()
        ped_boxes = table[table['[CameraBoxComponent].type'] == TYPE_PEDESTRIAN]

        orig_width, orig_height = orig_size
        boxes = []
        for _, row in ped_boxes.iterrows():
            x = row['[CameraBoxComponent].box.center.x'] / 1920 * orig_width
            y = row['[CameraBoxComponent].box.center.y'] / 1280 * orig_height
            w = row['[CameraBoxComponent].box.size.x'] / 1920 * orig_width
            h = row['[CameraBoxComponent].box.size.y'] / 1280 * orig_height
            if 0 < w < 300 and 0 < h < 300:
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
