import pyarrow.parquet as pq
import os

print("Starting minimal Waymo .parquet reader test...")

# Use a known working file (update this to a valid one if needed)
scene_id = "10017090168044687777_6380_000_6400_000"
base_dir = "/home/anthonyde/waymo_data/training"
lidar_path = os.path.join(base_dir, "lidar", f"{scene_id}.parquet")

print(f"Reading: {lidar_path}")

try:
    table = pq.read_table(lidar_path)
    df = table.to_pandas()
    print("✅ Successfully loaded parquet file.")
    print("DataFrame shape:", df.shape)
    print("Columns:", df.columns.tolist())
except Exception as e:
    print("❌ Failed to load parquet file:", e)
