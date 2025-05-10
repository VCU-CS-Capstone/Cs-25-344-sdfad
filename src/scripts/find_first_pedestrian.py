import os
import pyarrow.parquet as pq

# Attempt to use enum, fallback if unavailable
try:
    from waymo_open_dataset import label_pb2
    TYPE_PEDESTRIAN = label_pb2.Label.Type.Value("TYPE_PEDESTRIAN")
    CAMERA_FRONT = label_pb2.CameraName.Value("FRONT")
except:
    TYPE_PEDESTRIAN = 1
    CAMERA_FRONT = 1

root_dir = "/home/cs-25-344/waymo_data/training"
camera_box_dir = os.path.join(root_dir, "camera_box")

print("🔍 Scanning for pedestrian samples in FRONT camera...")

for fname in sorted(os.listdir(camera_box_dir)):
    if not fname.endswith(".parquet"):
        continue

    base_name = fname[:-8]
    path = os.path.join(camera_box_dir, fname)

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
        print(f"✅ Found pedestrian in: {base_name}")
        break
else:
    print("❌ No pedestrian found in any file.")
