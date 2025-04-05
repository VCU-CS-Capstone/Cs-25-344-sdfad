import os
import pyarrow.parquet as pq
from collections import Counter

# ==== CONFIG ====
ROOT_DIR = "/home/cs-25-344/waymo_data/training/camera_box"
NUM_FILES_TO_CHECK = 50  # Adjust this number as needed
# ===============

def get_base_filenames(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".parquet"))

def inspect_types(camera_box_dir, max_files=50):
    type_counter = Counter()
    files = get_base_filenames(camera_box_dir)[:max_files]
    
    print(f"🔍 Inspecting {len(files)} camera_box files...")

    for fname in files:
        path = os.path.join(camera_box_dir, fname)
        try:
            table = pq.read_table(path).to_pandas()
            if '[CameraBoxComponent].type' in table.columns:
                types = table['[CameraBoxComponent].type']
                type_counter.update(types)
            else:
                print(f"⚠️ No type column found in: {fname}")
        except Exception as e:
            print(f"❌ Failed to read {fname}: {e}")

    print("\n📊 Unique types and counts found:")
    for t, count in type_counter.items():
        print(f"  {repr(t):<25} → {count} times")

if __name__ == "__main__":
    inspect_types(ROOT_DIR, NUM_FILES_TO_CHECK)
