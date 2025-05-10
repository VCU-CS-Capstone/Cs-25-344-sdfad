import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=256):
        super(PointNetEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 4, N)
        x = self.encoder(x)   # (B, output_dim, N)
        x = torch.max(x, 2)[0]  # (B, output_dim)
        return x


class FusionModel(nn.Module):
    def __init__(self, num_classes=2, lidar_feature_dim=256, mode="fusion"):
        super(FusionModel, self).__init__()
        self.mode = mode
        self.rgb_encoder = EfficientNet.from_pretrained("efficientnet-b0")
        self.rgb_features = self.rgb_encoder._fc.in_features
        self.rgb_encoder._fc = nn.Identity()

        self.lidar_encoder = PointNetEncoder(input_dim=4, output_dim=lidar_feature_dim)
        self.lidar_features = lidar_feature_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(self.rgb_features + self.lidar_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Outputs: [x, y, w, h, confidence]
        )

    def forward(self, rgb_img, lidar_points):
        if self.mode == "camera":
            rgb_feat = self.rgb_encoder(rgb_img)
            lidar_feat = torch.zeros((rgb_feat.size(0), self.lidar_features), device=rgb_feat.device)
        elif self.mode == "lidar":
            rgb_feat = torch.zeros((lidar_points.size(0), self.rgb_features), device=lidar_points.device)
            lidar_feat = self.lidar_encoder(lidar_points)
        else:  # fusion
            rgb_feat = self.rgb_encoder(rgb_img)
            lidar_feat = self.lidar_encoder(lidar_points)

        combined = torch.cat([rgb_feat, lidar_feat], dim=1)
        output = self.fusion_head(combined)
        box_coords = output[:, :4]
        confidence = torch.sigmoid(output[:, 4])
        return box_coords, confidence


if __name__ == "__main__":
    dummy_rgb = torch.randn(2, 3, 224, 224)
    dummy_lidar = torch.randn(2, 100000, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(num_classes=2, mode="fusion").to(device)
    dummy_rgb = dummy_rgb.to(device)
    dummy_lidar = dummy_lidar.to(device)

    with torch.no_grad():
        boxes, conf = model(dummy_rgb, dummy_lidar)

    print("✅ Forward pass successful!")
    print(f"Box output shape: {boxes.shape}")
    print(f"Confidence shape: {conf.shape}")
