import struct

# Read the original binary PCD file
pcd_file = "velo0000.pcd"
ascii_pcd_file = "velo0000_ascii_fixed.pcd"

with open(pcd_file, "rb") as f:
    raw_data = f.read()

# Extract header (before "DATA binary_compressed")
header_end = raw_data.find(b"DATA binary_compressed") + len(b"DATA binary_compressed")
header = raw_data[:header_end].decode("latin1")  # Convert to string

# Extract binary data
binary_data = raw_data[header_end + 1:]  # Skip newline after "DATA binary_compressed"

# Parse header for number of points and field sizes
lines = header.split("\n")
point_count = None
point_size = 0
field_sizes = []

for line in lines:
    if line.startswith("POINTS"):
        point_count = int(line.split(" ")[1])
    elif line.startswith("SIZE"):
        field_sizes = list(map(int, line.split(" ")[1:]))  # Get sizes of each field
        point_size = sum(field_sizes)  # Calculate total bytes per point

if point_count is None or point_size == 0:
    print("❌ Could not determine POINTS or field sizes from PCD header.")
    exit()

# Recalculate number of points based on actual binary data size
actual_point_count = len(binary_data) // point_size

if actual_point_count != point_count:
    print(f"⚠️ Warning: Header says {point_count} points, but binary data suggests {actual_point_count} points.")
    point_count = actual_point_count  # Use corrected count

# Convert binary data to ASCII
points = []
offset = 0

for _ in range(point_count):
    point_values = []
    for i, size in enumerate(field_sizes):
        if size == 4:  # Float values (ensure little-endian format)
            value = struct.unpack("<f", binary_data[offset:offset+4])[0]
        elif size == 2:  # Unsigned short values (ensure little-endian format)
            value = struct.unpack("<H", binary_data[offset:offset+2])[0]
        else:
            value = None  # Ignore unexpected sizes
        point_values.append(value)
        offset += size

    points.append(" ".join(map(str, point_values)))

# Convert to ASCII and save
header = header.replace("DATA binary_compressed", "DATA ascii")  # Modify header
with open(ascii_pcd_file, "w") as f:
    f.write(header + "\n")
    f.write("\n".join(points) + "\n")

print(f"✅ Saved fixed ASCII PCD: {ascii_pcd_file}")
