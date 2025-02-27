import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.adaptive_pool(x)
        return x

class FusionModel(nn.Module):
    def __init__(self, num_classes=2, lidar_channels=4):
        super(FusionModel, self).__init__()
        self.lidar_encoder = LiDAREncoder(in_channels=lidar_channels)
        self.rgb_encoder = EfficientNet.from_pretrained("efficientnet-b0")
        self.rgb_features = self.rgb_encoder._fc.in_features
        self.lidar_features = 256 * 8 * 8  
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
        lidar_data = lidar_data.permute(0, 2, 3, 1)  
        if lidar_data.shape[1] != 4:
            raise ValueError(f"LiDAR data must have 4 channels, but got {lidar_data.shape[1]} channels")
        
        rgb_features = self.rgb_encoder(rgb_img).view(rgb_img.size(0), -1)
        lidar_features = self.lidar_encoder(lidar_data).view(lidar_data.size(0), -1)

        combined_features = torch.cat((rgb_features, lidar_features), dim=1)
        return self.fusion(combined_features)
