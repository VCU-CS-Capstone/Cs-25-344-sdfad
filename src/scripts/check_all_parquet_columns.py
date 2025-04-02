import pyarrow.parquet as pq
import os

# Root folder and test ID
root = "/home/cs-25-344/waymo_data/training"
test_id = "9985243312780923024_3049_720_3069_720"

# List of folders to inspect
folders = [
    "camera_image",
    "lidar",
    "camera_box",
    "lidar_box",
    "camera_calibration",
    "lidar_calibration"
]

# Inspect columns in each folder
for folder in folders:
    path = os.path.join(root, folder, test_id + ".parquet")
    print(f"\n🔍 Inspecting columns in: {folder}/")
    try:
        table = pq.read_table(path).to_pandas()
        print("✅ Columns found:")
        print(table.columns)
    except Exception as e:
        print("❌ Failed to read or parse:")
        print(e)

# Additional check for LiDAR type (range_image_return1)
print(f"\n🔍 Inspecting LiDAR range image type for: {test_id}")
lidar_path = os.path.join(root, 'lidar', test_id + '.parquet')
try:
    table = pq.read_table(lidar_path).to_pandas()
    entry = table['[LiDARComponent].range_image_return1.values'].iloc[0]
    print("TYPE:", type(entry))
    print("LENGTH:", len(entry))
except Exception as e:
    print("❌ Failed to load LiDAR data or inspect type:")
    print(e)
