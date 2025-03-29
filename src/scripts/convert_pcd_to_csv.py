import struct
import numpy as np
import pandas as pd

def parse_pcd_header(pcd_file):
    """ Reads the header of a PCD file and extracts field names and types. """
    header = []
    with open(pcd_file, 'rb') as f:
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            header.append(line)
            if line.startswith('DATA'):
                break
    return header

def convert_binary_pcd_to_csv(pcd_file, csv_file):
    """ Converts a binary PCD file to CSV format. """
    header = parse_pcd_header(pcd_file)

    # Extract number of points
    points_line = next(line for line in header if line.startswith('POINTS'))
    num_points = int(points_line.split()[1])

    # Extract fields (x, y, z, intensity, etc.)
    fields_line = next(line for line in header if line.startswith('FIELDS'))
    fields = fields_line.split()[1:]

    # Extract size per field
    size_line = next(line for line in header if line.startswith('SIZE'))
    sizes = list(map(int, size_line.split()[1:]))

    # Extract types (F = float, U = unsigned int)
    type_line = next(line for line in header if line.startswith('TYPE'))
    types = type_line.split()[1:]

    # Find binary data start position
    with open(pcd_file, 'rb') as f:
        raw_data = f.read()

    # Locate the start of binary data
    data_start = raw_data.index(b'DATA binary') + len(b'DATA binary\n')
    binary_data = raw_data[data_start:]

    # Define struct format dynamically
    format_map = {'F': 'f', 'U': 'H', 'I': 'I'}
    struct_format = ''.join(format_map[t] for t in types)
    struct_size = struct.calcsize(struct_format)

    # Ensure buffer size is correct
    if len(binary_data) % struct_size != 0:
        print(f"❌ ERROR: Binary data size ({len(binary_data)}) is not a multiple of struct size ({struct_size}).")
        return

    # Read points from binary data
    points = np.array([struct.unpack(struct_format, binary_data[i:i+struct_size]) 
                        for i in range(0, len(binary_data), struct_size)])

    # Save to CSV
    df = pd.DataFrame(points, columns=fields)
    df.to_csv(csv_file, index=False)

    print(f"✅ Successfully converted {pcd_file} -> {csv_file} ({num_points} points)")

# ----------------------
# 🔹 Command-Line Usage
# ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a binary PCD file to CSV.")
    parser.add_argument("pcd_file", type=str, help="Path to the .pcd file")
    parser.add_argument("csv_file", type=str, help="Path to save the .csv file")
    args = parser.parse_args()

    convert_binary_pcd_to_csv(args.pcd_file, args.csv_file)
