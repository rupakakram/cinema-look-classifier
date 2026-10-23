from PIL import Image
import numpy as np

files = ["cat.jpeg", "Rupak.jpeg", "Evening.jpeg"]

for f in files:
    img = Image.open(f).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    brightness = arr.mean()
    saturation = arr.std(axis=2).mean()
    print(f"{f}: brightness={brightness:.4f}, saturation={saturation:.4f}")
