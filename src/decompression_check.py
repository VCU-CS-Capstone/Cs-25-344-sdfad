import open3d as o3d

# Load the .pcd file
pcd = o3d.io.read_point_cloud("velo0000.pcd")

# Save in an uncompressed format
o3d.io.write_point_cloud("fixed_velo0000.pcd", pcd, write_ascii=False)
print("✅ Converted file saved as fixed_velo0000.pcd")
