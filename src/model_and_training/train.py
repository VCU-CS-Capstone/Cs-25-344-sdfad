# Usage:
# python train.py --epochs 10 --batch_size 4 --modality fusion --debug_interval 25
# --epochs: number of training epochs (default=5)
# --batch_size: number of samples per batch (default=8)
# --modality: sensor input mode (fusion, camera, or lidar)
# --debug_interval: show debug logs every N batches (0 disables)

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import argparse
from torch.utils.data import DataLoader
from model import FusionModel
from dataset import WaymoFusionDataset
from tqdm import tqdm

def find_nearest_pedestrian_box(pred_box, gt_boxes):
    if gt_boxes.size(0) == 0:
        return torch.zeros(4, device=pred_box.device), False
    pred_center = pred_box[:2]
    gt_centers = gt_boxes[:, :2]
    distances = torch.norm(gt_centers - pred_center, dim=1)
    nearest_idx = torch.argmin(distances)
    return gt_boxes[nearest_idx], True

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2
    x2_min, y2_min, x2_max, y2_max = x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def train_one_epoch(model, dataloader, optimizer, device, debug_interval=0):
    model.train()
    total_loss = 0.0
    num_zero_preds = 0
    total_preds = 0

    for batch_idx, (rgb_batch, lidar_batch, target_batch) in enumerate(tqdm(dataloader, desc="Training")):
        print(f"Training batch {batch_idx+1}/{len(dataloader)}")
        rgb = torch.stack(rgb_batch).to(device)
        lidar = torch.stack(lidar_batch).to(device)
        targets = list(target_batch)

        gt_boxes = [t['boxes'].to(device) for t in targets]

        optimizer.zero_grad()
        pred_boxes, confidences = model(rgb, lidar)

        losses = []
        conf_losses = []

        for i in range(pred_boxes.size(0)):
            matched_gt, found = find_nearest_pedestrian_box(pred_boxes[i], gt_boxes[i])
            if found:
                box_loss = nn.functional.smooth_l1_loss(pred_boxes[i], matched_gt)
                conf_loss = nn.functional.binary_cross_entropy(confidences[i], torch.tensor(1.0, device=device))
                losses.append(box_loss)
                conf_losses.append(conf_loss)
            else:
                conf_loss = nn.functional.binary_cross_entropy(confidences[i], torch.tensor(0.0, device=device))
                conf_losses.append(conf_loss)

        if losses or conf_losses:
            total = 0
            if losses:
                total += torch.stack(losses).mean()
            if conf_losses:
                total += torch.stack(conf_losses).mean()
            total.backward()
            optimizer.step()
            total_loss += total.item()

        zero_preds = (pred_boxes == 0).all(dim=1).sum().item()
        num_zero_preds += zero_preds
        total_preds += pred_boxes.shape[0]

        if debug_interval > 0 and (batch_idx + 1) % debug_interval == 0:
            logging.info(f"[DEBUG] Batch {batch_idx+1}: Zero preds = {zero_preds}/{pred_boxes.shape[0]}")
            logging.info(f"[DEBUG] Pred sample: {pred_boxes[0].tolist()} | Confidence: {confidences[0].item():.4f}")

    logging.info(f"Total zero predictions: {num_zero_preds}/{total_preds}")
    print(f"Completed training epoch with {total_preds} predictions.")
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_alarm_file(pred_boxes, confidences, targets, out_path):
    import pandas as pd
    print(f"Saving alarm file to {out_path} with {len(pred_boxes)} predictions")
    rows = []
    for pred, conf, tgt in zip(pred_boxes, confidences, targets):
        matched = 0
        for gt in tgt['boxes']:
            iou = compute_iou(pred, gt)
            if iou >= 0.5:
                matched = 1
                break
        row = {
            'pred_x': pred[0].item(), 'pred_y': pred[1].item(),
            'pred_w': pred[2].item(), 'pred_h': pred[3].item(),
            'confidence': conf.item(),
            'num_gt_boxes': len(tgt['boxes']),
            'matched_gt': matched
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--modality", type=str, choices=["fusion", "camera", "lidar"], default="fusion", help="Sensor modality to use")
    parser.add_argument("--debug_interval", type=int, default=0, help="Print debug info every N batches (0 to disable)")
    parser.add_argument("--dataset_limit", type=int, default=None,
                        help="Limit number of dataset samples (for fast testing)")
    args = parser.parse_args()

    output_dir = "/home/cs-25-344/Cs-25-344-sdfad/src/outputs"
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "train_log.txt")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_path),
                                  logging.StreamHandler()])

    root_dir = "/home/cs-25-344/waymo_data/training"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading dataset...")
    dataset = WaymoFusionDataset(root_dir=root_dir, limit=args.dataset_limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: list(zip(*x)))
    print(f"✅ Dataset loaded with {len(dataset)} samples")

    print("Initializing model...")
    model = FusionModel(num_classes=4, mode=args.modality).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        avg_loss = train_one_epoch(model, dataloader, optimizer, device, debug_interval=args.debug_interval)
        logging.info(f"Avg Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

    # Save the trained model weights
    model_path = os.path.join(output_dir, "fusion_model_weights.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"✅ Model weights saved to {model_path}")

    print("Evaluating model...")
    model.eval()
    pred_boxes = []
    confidences = []
    all_targets = []
    with torch.no_grad():
        for rgb_batch, lidar_batch, target_batch in tqdm(dataloader, desc="Evaluating"):
            rgb = torch.stack(rgb_batch).to(device)
            lidar = torch.stack(lidar_batch).to(device)
            targets = list(target_batch)
            box, conf = model(rgb, lidar)
            pred_boxes.extend(box.cpu())
            confidences.extend(conf.cpu())
            all_targets.extend(targets)

    alarm_path = os.path.join(output_dir, "alarm_file.csv")
    save_alarm_file(pred_boxes, confidences, all_targets, out_path=alarm_path)
    logging.info("✅ Training and alarm file generation complete.")
    print("✅ Script finished.")
