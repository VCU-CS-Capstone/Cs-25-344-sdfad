from pypcd import pypcd

# Open in binary mode
with open("velo0000.pcd", "rb") as f:
    raw_data = f.read()

# Replace "DATA binary_compressed" with "DATA ascii" safely
raw_data = raw_data.replace(b"DATA binary_compressed", b"DATA ascii")

# Save a modified version
with open("velo0000_forced_ascii.pcd", "wb") as f:
    f.write(raw_data)

# Try loading with pypcd
pc = pypcd.PointCloud.from_path("velo0000_forced_ascii.pcd")

# Print some info
print(f"✅ Loaded {pc.pc_data.shape[0]} points using pypcd")
print(f"📊 First 5 points:\n{pc.pc_data[['x', 'y', 'z']][:5]}")
