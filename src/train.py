import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import open3d as o3d
import numpy as np
import pandas as pd
import glob
import os
import scipy.io  # For reading .mat files
from efficientnet_pytorch import EfficientNet
import logging
import random

# --------------------- 
# RGB Image Preprocessing
# ---------------------
def load_rgb_image(img_path):
    """Loads an RGB image, resizes, normalizes, and converts to a tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for EfficientNet
        transforms.ToTensor(),          # Convert to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)  # Returns Tensor of shape [3, 224, 224]

# ---------------------
# Fix LiDAR Data Preprocessing (Ensure 4 Channels)
# ---------------------
def load_lidar_data(pcd_path, max_points=4096):
    """Loads a LiDAR .pcd file, ensures it has 4 channels (x, y, z, intensity) and correct shape for Conv2D."""
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)  # Extract (x, y, z)

    # Create a fake intensity channel (if missing)
    if points.shape[1] == 3:  # If intensity is missing, add a zero-intensity channel
        intensity = np.zeros((points.shape[0], 1))  
        points = np.hstack((points, intensity))  # Now shape [N, 4]

    # Pad/truncate to maintain fixed input size
    if points.shape[0] > max_points:
        points = points[:max_points]  # Truncate excess points
    else:
        padding = np.zeros((max_points - points.shape[0], 4))  # Pad with zeros
        points = np.vstack((points, padding))

    # Convert to PyTorch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32)  # Shape [4096, 4]

    # Reshape for Conv2D input: [batch, channels, height, width]
    lidar_tensor = points_tensor.T.unsqueeze(0)  # Shape [1, 4, 4096]

    return lidar_tensor

# ---------------------
# Load Labels from .mat Files
# ---------------------
def load_annotation(mat_path):
    """Loads the annotation from a .mat file and returns the pedestrian's class label."""
    mat_data = scipy.io.loadmat(mat_path)

    # Extract pedestrian detection data
    detection = mat_data.get("detection", None)  # Ensure key exists
    if detection is None or detection.shape[0] == 0:
        return torch.tensor(0, dtype=torch.long)  # No pedestrian detected → Class 0 (background)

    # Extract 'type' column (0 = pedestrian)
    label = int(detection[0, 5])  # 'type' column in .mat file
    return torch.tensor(1, dtype=torch.long) if label == 0 else torch.tensor(0, dtype=torch.long)  # 1 = pedestrian, 0 = background

# ---------------------
# Match Images and LiDAR Data Using Timestamps
# ---------------------
def match_timestamps(img_timestamps, lidar_timestamps):
    """
    Matches LiDAR frames to the closest image timestamps.
    Returns a list of (image_index, lidar_index) pairs.
    """
    matched_indices = []
    for img_idx, img_ts in enumerate(img_timestamps):
        closest_lidar_idx = np.argmin(np.abs(lidar_timestamps - img_ts))
        matched_indices.append((img_idx, closest_lidar_idx))
    return matched_indices

# ---------------------
# Dataset Class (Matches RGB, LiDAR & Annotations)
# ---------------------
class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        """Initializes the dataset, loads timestamps, and finds matching image-LiDAR pairs."""
        self.img_dir = os.path.join(root_dir, "img")
        self.lidar_dir = os.path.join(root_dir, "velo")
        self.anno_dir = os.path.join(root_dir, "annotations")  # Annotations folder

        # Load file paths
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_dir, "*.pcd")))
        self.anno_files = sorted(glob.glob(os.path.join(self.anno_dir, "detect*.mat")))

        # Load timestamps
        img_timestamps = pd.read_csv(os.path.join(root_dir, "imgtimestamps.csv")).values[:, 2]
        lidar_timestamps = pd.read_csv(os.path.join(root_dir, "velotimestamps.csv")).values[:, 2]

        # Match timestamps
        matched_pairs = match_timestamps(img_timestamps, lidar_timestamps)

        # Store only matched file paths
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

# ---------------------
# Define Model (Fusion of RGB + LiDAR)
# ---------------------
class LiDAREncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(LiDAREncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Make output fixed size

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.adaptive_pool(x)  # Apply adaptive pooling to ensure fixed size
        return x

class FusionModel(nn.Module):
    def __init__(self, num_classes=2, lidar_channels=4):
        super(FusionModel, self).__init__()
        self.lidar_encoder = LiDAREncoder(in_channels=lidar_channels)
        self.rgb_encoder = EfficientNet.from_pretrained("efficientnet-b0")
        self.rgb_features = self.rgb_encoder._fc.in_features
        self.lidar_features = 256 * 8 * 8  # After adaptive pooling
        self.rgb_encoder._fc = nn.Identity()
        self.fusion = nn.Sequential(
            nn.Linear(self.rgb_features + self.lidar_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb_img, lidar_data):
        # Reorder the LiDAR data dimensions to [batch_size, 4, 4096, 1] using permute
        lidar_data = lidar_data.permute(0, 2, 3, 1)  # Change order to [batch, 4, 4096, 1]

        # Ensure that LiDAR data has the correct shape: [batch_size, 4, 4096, 1]
        if lidar_data.shape[1] != 4:
            raise ValueError(f"LiDAR data must have 4 channels, but got {lidar_data.shape[1]} channels")

        # Pass RGB data through EfficientNet encoder
        rgb_features = self.rgb_encoder(rgb_img).view(rgb_img.size(0), -1)

        # Pass LiDAR data through LiDAR encoder
        lidar_features = self.lidar_encoder(lidar_data).view(lidar_data.size(0), -1)

        # Concatenate RGB and LiDAR features and pass through the fusion layers
        combined_features = torch.cat((rgb_features, lidar_features), dim=1)
        return self.fusion(combined_features)

# ---------------------
# Training Function
# ---------------------
def train_fusion_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        epoch_loss = 0  # Track loss for the epoch
        for i, (rgb_imgs, lidar_data, labels) in enumerate(train_loader):
            rgb_imgs = rgb_imgs.to(device)
            lidar_data = lidar_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb_imgs, lidar_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Output random (or pseudo-random) tensors every 100 batches
            if (i + 1) % 100 == 0:
                random_idx = random.randint(0, len(train_loader) - 1)
                random_rgb, random_lidar, _ = train_loader.dataset[random_idx]
                logging.info(f"Random RGB Tensor (batch {i+1}): {random_rgb.shape}")
                logging.info(f"Random LiDAR Tensor (batch {i+1}): {random_lidar.shape}")

        # Log average loss for the epoch
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss / len(train_loader):.4f}")

# ---------------------
# Main Training Script
# ---------------------
def main():
    # Setup logging to file and console
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s', 
                        handlers=[
                            logging.FileHandler("train_log.txt"),
                            logging.StreamHandler()
                        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    train_dataset = FusionDataset(root_dir=r"C:\Users\Danth\Downloads\Copy of _2019-02-09-16-10-44\_2019-02-09-16-10-44")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_fusion_model(model, train_loader, criterion, optimizer, device)

if __name__ == "__main__":
    main()
