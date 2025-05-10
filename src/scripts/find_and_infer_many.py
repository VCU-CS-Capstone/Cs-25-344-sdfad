
import os
import argparse
import subprocess
import pyarrow.parquet as pq

# === Config ===
ROOT_DIR = "/home/cs-25-344/waymo_data/training"
CAMERA_BOX_DIR = os.path.join(ROOT_DIR, "camera_box")
INFER_SCRIPT = "/home/cs-25-344/Cs-25-344-sdfad/src/model_and_training/mass_inference.py"
OUTPUT_DIR = "/home/cs-25-344/Cs-25-344-sdfad/src/outputs"

try:
    from waymo_open_dataset import label_pb2
    TYPE_PEDESTRIAN = label_pb2.Label.Type.Value("TYPE_PEDESTRIAN")
    CAMERA_FRONT = label_pb2.CameraName.Value("FRONT")
except:
    TYPE_PEDESTRIAN = 1
    CAMERA_FRONT = 1

# === Args ===
parser = argparse.ArgumentParser()
parser.add_argument("--max", type=int, default=5, help="Number of outputs to generate")
args = parser.parse_args()

print(f"🔍 Scanning for up to {args.max} samples with pedestrians in FRONT camera...")
matches = []

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
        print(f"✅ Match: {base_name}")
        matches.append(base_name)

    if len(matches) >= args.max:
        break

if not matches:
    print("❌ No samples found.")
    exit(1)

# === Run inference for each match ===
for base_name in matches:
    output_img = os.path.join(OUTPUT_DIR, f"infer_{base_name}.png")
    print(f"🚀 Inferring {base_name}...")
    subprocess.run([
        "python", INFER_SCRIPT,
        "--single-sample", base_name,
        "--out", output_img
    ], check=True)
    print(f"✅ Saved to {output_img}")

print("🏁 All done!")
