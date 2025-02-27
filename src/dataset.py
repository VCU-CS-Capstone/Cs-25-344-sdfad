import torch
import torchvision.transforms as transforms
from PIL import Image
import open3d as o3d
import numpy as np
import pandas as pd
import os
import glob
import scipy.io

def load_rgb_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def load_lidar_data(pcd_path, max_points=4096):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  

    if points.shape[1] == 3:
        intensity = np.zeros((points.shape[0], 1))  
        points = np.hstack((points, intensity))

    if points.shape[0] > max_points:
        points = points[:max_points]  
    else:
        padding = np.zeros((max_points - points.shape[0], 4))  
        points = np.vstack((points, padding))

    points_tensor = torch.tensor(points, dtype=torch.float32)  
    lidar_tensor = points_tensor.T.unsqueeze(0)  

    return lidar_tensor

def load_annotation(mat_path):
    mat_data = scipy.io.loadmat(mat_path)
    detection = mat_data.get("detection", None)  
    if detection is None or detection.shape[0] == 0:
        return torch.tensor(0, dtype=torch.long)  

    label = int(detection[0, 5])  
    return torch.tensor(1, dtype=torch.long) if label == 0 else torch.tensor(0, dtype=torch.long)

def match_timestamps(img_timestamps, lidar_timestamps):
    matched_indices = []
    for img_idx, img_ts in enumerate(img_timestamps):
        closest_lidar_idx = np.argmin(np.abs(lidar_timestamps - img_ts))
        matched_indices.append((img_idx, closest_lidar_idx))
    return matched_indices

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "img")
        self.lidar_dir = os.path.join(root_dir, "velo")
        self.anno_dir = os.path.join(root_dir, "annotations")  

        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.pcd")))
        self.anno_files = sorted(glob.glob(os.path.join(self.anno_dir, "detect*.mat")))

        img_timestamps = pd.read_csv(os.path.join(root_dir, "imgtimestamps.csv")).values[:, 2]
        lidar_timestamps = pd.read_csv(os.path.join(root_dir, "velotimestamps.csv")).values[:, 2]
        matched_pairs = match_timestamps(img_timestamps, lidar_timestamps)

        self.matched_img_files = [self.img_files[i] for i, j in matched_pairs]
        self.matched_lidar_files = [self.lidar_files[j] for i, j in matched_pairs]
        self.matched_anno_files = [self.anno_files[i] for i, j in matched_pairs]

    def __len__(self):
        return len(self.matched_img_files)

    def __getitem__(self, idx):
        rgb_img = load_rgb_image(self.matched_img_files[idx])
        lidar_data = load_lidar_data(self.matched_lidar_files[idx])
        label = load_annotation(self.matched_anno_files[idx])
        return rgb_img, lidar_data, label
