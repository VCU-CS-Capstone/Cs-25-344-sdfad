# Usage:
# python inference_visual.py --single-sample <base_filename> --out <output_image_path_prefix>

import os
import io
import argparse
import torch
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from torchvision import transforms
from model import FusionModel

# Robust pedestrian and camera front type detection
try:
    from waymo_open_dataset import label_pb2
    TYPE_PEDESTRIAN = getattr(label_pb2.Label.Type, "TYPE_PEDESTRIAN", 2)
    CAMERA_FRONT = 1  # Set directly to 1 since label_pb2.CameraName may not exist
except Exception:
    TYPE_PEDESTRIAN = 2
    CAMERA_FRONT = 1

def load_camera_front_image_and_boxes(root_dir, base_name, transform):
    cam_path = os.path.join(root_dir, 'camera_image', base_name + '.parquet')
    box_path = os.path.join(root_dir, 'camera_box', base_name + '.parquet')

    cam_table = pq.read_table(cam_path).to_pandas()
    box_table = pq.read_table(box_path).to_pandas()

    rows = cam_table[cam_table['key.camera_name'] == CAMERA_FRONT]
    if rows.empty:
        raise ValueError("No CAMERA_FRONT image found for sample")

    image_bytes = rows.iloc[0]['[CameraImageComponent].image']
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_width, orig_height = img.size
    tensor = transform(img)

    box_table["[CameraBoxComponent].type"] = box_table["[CameraBoxComponent].type"].astype(int)
    box_table["key.camera_name"] = box_table["key.camera_name"].astype(int)

    cam_filtered = box_table[box_table['key.camera_name'] == CAMERA_FRONT]
    ped_boxes = cam_filtered[cam_filtered['[CameraBoxComponent].type'] == TYPE_PEDESTRIAN]

    score_column = '[CameraBoxComponent].score'
    if score_column in ped_boxes.columns:
        ped_boxes = ped_boxes[ped_boxes[score_column] > 0.3]

    boxes = []
    for _, row in ped_boxes.iterrows():
        x = row['[CameraBoxComponent].box.center.x'] / 1920 * orig_width
        y = row['[CameraBoxComponent].box.center.y'] / 1280 * orig_height
        w = row['[CameraBoxComponent].box.size.x'] / 1920 * orig_width
        h = row['[CameraBoxComponent].box.size.y'] / 1280 * orig_height
        if 0 < w < 300 and 0 < h < 300:
            boxes.append([x, y, w, h])

    return img, tensor, torch.tensor(boxes, dtype=torch.float32)

def load_lidar_tensor(root_dir, base_name):
    lidar_path = os.path.join(root_dir, 'lidar', base_name + '.parquet')
    lidar_table = pq.read_table(lidar_path).to_pandas()
    raw_data = lidar_table['[LiDARComponent].range_image_return1.values'].iloc[0]
    points = np.array(raw_data).reshape(-1, 4)
    max_points = 100000
    if points.shape[0] > max_points:
        points = points[:max_points]
    else:
        padding = np.zeros((max_points - points.shape[0], 4))
        points = np.vstack([points, padding])
    return torch.tensor(points, dtype=torch.float32)

def draw_boxes(img, pred_box, gt_boxes, confidence, skip_pred=False, skip_gt=False):
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if not skip_gt:
        for box in gt_boxes:
            x, y, w, h = box
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = max(0, w)
            h = max(0, h)
            draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline="green", width=4)

    if not skip_pred and not torch.all(pred_box == 0):
        x, y, w, h = pred_box
        x = max(0, min(x, width))
        y = max(0, min(y, height))
        draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline="red", width=2)
        draw.text((x + w / 2 + 5, y - h / 2), f"conf: {confidence:.2f}", fill="red")

    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-sample", type=str, required=True, help="Base filename of the sample")
    parser.add_argument("--out", type=str, default="viz_output", help="Prefix for output image files")
    parser.add_argument("--no-infer", action="store_true", help="Skip model prediction and show GT boxes only")
    parser.add_argument("--no-boxes", action="store_true", help="Don't draw any boxes, just output raw images")
    args = parser.parse_args()

    base_name = args.single_sample
    root_dir = "/home/cs-25-344/waymo_data/training"
    output_path = args.out + "_FRONT.png"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"🔍 Loading sample: {base_name}")
    img, rgb_tensor, gt_boxes = load_camera_front_image_and_boxes(root_dir, base_name, transform)
    lidar_tensor = load_lidar_tensor(root_dir, base_name)

    if not args.no_infer:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FusionModel(num_classes=4, mode="fusion")
        weights_path = "/home/cs-25-344/Cs-25-344-sdfad/src/outputs/fusion_model_weights.pth"
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval().to(device)

        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
        lidar_tensor = lidar_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_box, conf = model(rgb_tensor, lidar_tensor)
        pred_box = pred_box[0].cpu()
        conf = conf[0].cpu()

        print(f"📦 Prediction: {pred_box.tolist()} | Confidence: {conf:.2f}")
    else:
        pred_box = torch.zeros(4)
        conf = torch.tensor(0.0)

    result_img = draw_boxes(img.copy(), pred_box, gt_boxes, conf,
                            skip_pred=args.no_infer or args.no_boxes,
                            skip_gt=args.no_boxes)
    result_img.save(output_path)
    print(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    main()
