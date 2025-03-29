print("Starting AVX test...")

try:
    import torch
    print("✅ torch imported successfully")
except Exception as e:
    print("❌ torch import failed:", e)

try:
    import torchvision
    print("✅ torchvision imported successfully")
except Exception as e:
    print("❌ torchvision import failed:", e)

try:
    import numpy
    print("✅ numpy imported successfully")
except Exception as e:
    print("❌ numpy import failed:", e)

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except Exception as e:
    print("❌ pandas import failed:", e)

try:
    import pyarrow
    print("✅ pyarrow imported successfully")
except Exception as e:
    print("❌ pyarrow import failed:", e)

try:
    import cv2
    print("✅ opencv (cv2) imported successfully")
except Exception as e:
    print("❌ cv2 import failed:", e)

try:
    import efficientnet_pytorch
    print("✅ efficientnet_pytorch imported successfully")
except Exception as e:
    print("❌ efficientnet_pytorch import failed:", e)
