from dataset import WaymoParquetDataset

print("Loading dataset...")
ds = WaymoParquetDataset(
    scene_ids=["10017090168044687777_6380_000_6400_000"],
    base_dir="/home/anthonyde/waymo_data/training"
)

print("Dataset length:", len(ds))
sample = ds[0]
print("Sample loaded:")
print("  - RGB shape:", sample[0].shape)
print("  - LiDAR shape:", sample[1].shape)
print("  - Label:", sample[2])
