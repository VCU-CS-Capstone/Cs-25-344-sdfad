import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import DataLoader, ConcatDataset
from model import FusionModel
from dataset import FusionDataset

def get_all_subfolders(root_dir):
    """Finds all subdirectories containing datasets within the main root directory."""
    return [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, subfolder))]

def train_fusion_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Training loop for the model."""
    model.train()
    for epoch in range(num_epochs):
        logging.info(f"Epoch [{epoch+1}/{num_epochs}]")
        epoch_loss = 0
        for rgb_imgs, lidar_data, labels in train_loader:
            rgb_imgs, lidar_data, labels = rgb_imgs.to(device), lidar_data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb_imgs, lidar_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s', 
                        handlers=[logging.FileHandler("train_log.txt"),
                                  logging.StreamHandler()])

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = FusionModel().to(device)

    # Define the parent directory containing multiple dataset folders
    parent_dir = r"C:\Users\Danth\Downloads\Copy of _2019-02-09-16-10-44"

    # Get all subfolders containing datasets
    dataset_folders = get_all_subfolders(parent_dir)

    # Load all datasets from subfolders
    datasets = [FusionDataset(root_dir=subfolder) for subfolder in dataset_folders]
    
    # Combine datasets if multiple exist
    if len(datasets) > 1:
        train_dataset = ConcatDataset(datasets)  # Merges all datasets
    else:
        train_dataset = datasets[0]

    # DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_fusion_model(model, train_loader, criterion, optimizer, device)
