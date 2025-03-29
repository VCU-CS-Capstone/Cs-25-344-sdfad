import pandas as pd

# Read the ASCII PCD file
with open("velo0000_ascii_fixed.pcd", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find where the data starts
data_start = None
for i, line in enumerate(lines):
    if line.strip() == "DATA ascii":
        data_start = i + 1  # Data starts on the next line
        break

if data_start is None:
    print("❌ Could not find 'DATA ascii' in the file.")
    exit()

# Extract column names from the header
for line in lines:
    if line.startswith("FIELDS"):
        fields = line.split()[1:]
        break

# Read the point cloud data
point_data = []
for line in lines[data_start:]:
    values = list(map(float, line.strip().split()))
    point_data.append(values)

# Convert to Pandas DataFrame
df = pd.DataFrame(point_data, columns=fields)

# Print some information
print(f"✅ Successfully loaded {len(df)} points from ASCII PCD")
print(f"📊 First 5 points:\n{df.head()}")
