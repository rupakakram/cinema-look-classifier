import os
import random
from PIL import Image
import numpy as np

FOLDERS = ["data/cinematic", "data/normal"]
SAMPLE_SIZE = 200

def analyze_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = random.sample(files, min(SAMPLE_SIZE, len(files)))

    file_sizes = []
    modes = set()
    brightness_vals = []
    saturation_vals = []
    aspect_ratios = set()

    for file_name in sample:
        path = os.path.join(folder, file_name)
        file_sizes.append(os.path.getsize(path))

        img = Image.open(path)
        modes.add(img.mode)
        aspect_ratios.add(round(img.size[0] / img.size[1], 2))

        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb).astype(np.float32) / 255.0

        brightness = np.mean(arr)
        brightness_vals.append(brightness)

        saturation = arr.std(axis=2).mean()
        saturation_vals.append(saturation)

    print(f"\n--- {folder} (sample of {len(sample)}) ---")
    print(f"File size (bytes): min={min(file_sizes)}, max={max(file_sizes)}, avg={int(np.mean(file_sizes))}")
    print(f"Color modes found: {modes}")
    print(f"Aspect ratios found: {aspect_ratios}")
    print(f"Avg brightness (0-1): {np.mean(brightness_vals):.4f}")
    print(f"Avg saturation proxy: {np.mean(saturation_vals):.4f}")

def main():
    random.seed(42)
    for folder in FOLDERS:
        if os.path.isdir(folder):
            analyze_folder(folder)
        else:
            print(f"Folder not found: {folder}")


if __name__ == "__main__":
    main()
