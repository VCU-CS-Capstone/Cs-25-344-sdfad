import numpy as np
import pandas as pd
import sys
import os
from pypcd import pypcd

def test_pypcd(pcd_file):
    """Test function to read and process a .pcd file using pypcd."""
    
    try:
        # Load the PCD file (forcing binary mode)
        print(f"📂 Loading PCD file: {pcd_file}")
        
        with open(pcd_file, "rb") as f:  # Open in binary mode to avoid encoding issues
            pc = pypcd.PointCloud.from_fileobj(f)

        # Extract XYZ coordinates (if available)
        required_fields = ['x', 'y', 'z']
        if not all(field in pc.fields for field in required_fields):
            print("❌ PCD file is missing required XYZ fields!")
            return

        # Convert to NumPy array
        points = np.vstack([pc.pc_data[field] for field in required_fields]).T  # Shape: (N, 3)

        # Print some basic stats
        print(f"✅ Loaded {points.shape[0]} points")
        print(f"📊 X range: {points[:, 0].min()} to {points[:, 0].max()}")
        print(f"📊 Y range: {points[:, 1].min()} to {points[:, 1].max()}")
        print(f"📊 Z range: {points[:, 2].min()} to {points[:, 2].max()}")

        # Save to CSV for manual inspection
        csv_file = pcd_file.replace(".pcd", ".csv")
        df = pd.DataFrame(points, columns=['x', 'y', 'z'])
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved extracted points to: {csv_file}")

    except Exception as e:
        print(f"❌ Error processing {pcd_file}: {e}")

# -------------------------
# 🔹 MAIN SCRIPT EXECUTION
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pypcd.py <input.pcd>")
        sys.exit(1)

    input_pcd = sys.argv[1]

    if not os.path.exists(input_pcd):
        print(f"❌ File not found: {input_pcd}")
        sys.exit(1)

    test_pypcd(input_pcd)
