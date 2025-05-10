
import os
import subprocess
import pyarrow.parquet as pq

# === Configuration ===
ROOT_DIR = "/home/cs-25-344/waymo_data/training"
CAMERA_BOX_DIR = os.path.join(ROOT_DIR, "camera_box")
INFER_SCRIPT = "/home/cs-25-344/Cs-25-344-sdfad/src/model_and_training/inference_visual_compare.py"
OUTPUT_DIR = "/home/cs-25-344/Cs-25-344-sdfad/src/outputs"
MODELA_PATH = "/home/cs-25-344/Cs-25-344-sdfad/src/outputs/saved/fusion_model_weights_2137.pth"
MODELB_PATH = "/home/cs-25-344/Cs-25-344-sdfad/src/outputs/saved/fusion_model_weights_2157_camera.pth"

# Use enum constants if available
try:
    from waymo_open_dataset import label_pb2
    TYPE_PEDESTRIAN = label_pb2.Label.Type.Value("TYPE_PEDESTRIAN")
    CAMERA_FRONT = label_pb2.CameraName.Value("FRONT")
except:
    TYPE_PEDESTRIAN = 1
    CAMERA_FRONT = 1

print("🔍 Scanning for a pedestrian sample in FRONT camera...")

sample_found = None
for fname in sorted(os.listdir(CAMERA_BOX_DIR)):
    if not fname.endswith(".parquet"):
        continue

    base_name = fname[:-8]
    path = os.path.join(CAMERA_BOX_DIR, fname)

    try:
        table = pq.read_table(path).to_pandas()
    except Exception as e:
        print(f"❌ Failed to read {fname}: {e}")
        continue

    filtered = table[
        (table["[CameraBoxComponent].type"] == TYPE_PEDESTRIAN) &
        (table["key.camera_name"] == CAMERA_FRONT)
    ]

    if not filtered.empty:
        sample_found = base_name
        print(f"✅ Found pedestrian in: {base_name}")
        break

if not sample_found:
    print("❌ No pedestrian found in any file.")
    exit(1)

# === Run inference script ===
output_img = os.path.join(OUTPUT_DIR, f"infer_{sample_found}.png")
print(f"🚀 Running inference on {sample_found}...")
cmd = [
    "python", INFER_SCRIPT,
    "--single-sample", sample_found,
    "--out", output_img
]
subprocess.run(cmd, check=True)
print(f"✅ Done! Output saved to {output_img}")
