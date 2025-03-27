import struct
import binascii

# Read the binary part of the PCD file
pcd_file = "velo0000.pcd"

with open(pcd_file, "rb") as f:
    raw_data = f.read()

# Find the start of the binary data (after the header)
header_end = raw_data.find(b"DATA binary_compressed") + len(b"DATA binary_compressed")
binary_data = raw_data[header_end:]

# Define the point size (18 bytes)
point_size = 18

# Number of points based on the file size
num_points = len(binary_data) // point_size
print(f"\n📊 Binary data contains {num_points} points (based on file size).")

# Read the first 3 points to check
for i in range(3):
    offset = i * point_size
    raw_point = binary_data[offset:offset+point_size]
    print(f"\n🔹 Point {i} raw bytes: {binascii.hexlify(raw_point)}")

# Check the first 3 points in Little-Endian (<f, <H)
print("\n🔹 Checking Little-Endian (<f, <H) for first 3 points")
for i in range(3):
    offset = i * point_size
    x = struct.unpack("<f", binary_data[offset:offset+4])[0]
    y = struct.unpack("<f", binary_data[offset+4:offset+8])[0]
    z = struct.unpack("<f", binary_data[offset+8:offset+12])[0]
    intensity = struct.unpack("<f", binary_data[offset+12:offset+16])[0]
    ring = struct.unpack("<H", binary_data[offset+16:offset+18])[0]  # Read the ring field

    print(f"Point {i}: x={x}, y={y}, z={z}, intensity={intensity}, ring={ring}")
