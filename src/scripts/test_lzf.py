import sys
import os
import lzf

def decompress_pcd(pcd_file):
    """Reads and decompresses a 'binary_compressed' PCD file."""
    try:
        with open(pcd_file, "rb") as f:
            raw_data = f.read().decode("latin1")  # Read entire file in binary mode

        # Split header and data
        header_end = raw_data.find("DATA binary_compressed") + len("DATA binary_compressed")
        if header_end == -1:
            print("❌ Could not find 'DATA binary_compressed' in PCD file")
            return None

        header = raw_data[:header_end].strip().split("\n")
        # Extract only the raw compressed binary data
        compressed_data = raw_data[header_end:].encode("latin1").lstrip(b'\n')  # Remove unexpected leading newline

        print(f"🔎 First 100 bytes of compressed data: {compressed_data[:100]}")


        print("🔹 Found header, extracting POINTS count...")

        # Extract the POINTS count
        point_count = None
        point_size = 0  # We will determine this from SIZE field
        for line in header:
            if line.startswith("POINTS"):
                point_count = int(line.split(" ")[1])
            elif line.startswith("SIZE"):
                sizes = list(map(int, line.split(" ")[1:]))  # Get sizes of each field
                point_size = sum(sizes)  # Sum up the bytes per point

        if point_count is None or point_size == 0:
            print("❌ Could not determine POINTS or field sizes from PCD header")
            return None

        expected_size = point_count * point_size  # Compute expected uncompressed size

        print(f"🔹 POINTS: {point_count}, Estimated decompressed size: {expected_size} bytes")

        # Decompress binary data
        # Print first 50 bytes of compressed data for debugging
        # Skip first 16 bytes and attempt decompression
        compressed_data_trimmed = compressed_data[16:]  # Adjust this number if needed

        try:
            decompressed_data = lzf.decompress(compressed_data_trimmed, expected_size)
            print("✅ Decompression successful!")
        except Exception as e:
            print(f"❌ Decompression still failed: {e}")




        # Save decompressed PCD
        decompressed_pcd = pcd_file.replace(".pcd", "_decompressed.pcd")
        with open(decompressed_pcd, "wb") as f:
            f.write("\n".join(header).encode("utf-8") + b"\n")
            f.write(decompressed_data)

        print(f"✅ Decompressed file saved as: {decompressed_pcd}")
        return decompressed_pcd

    except Exception as e:
        print(f"❌ Error during decompression: {e}")
        return None

# -------------------------
# 🔹 MAIN SCRIPT EXECUTION
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_lzf.py <input.pcd>")
        sys.exit(1)

    input_pcd = sys.argv[1]

    if not os.path.exists(input_pcd):
        print(f"❌ File not found: {input_pcd}")
        sys.exit(1)

    decompress_pcd(input_pcd)
