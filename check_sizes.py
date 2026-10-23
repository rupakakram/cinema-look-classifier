import os
from PIL import Image

for folder in ["data/cinematic", "data/normal"]:
    size = set()
    files = os.listdir(folder)[:20]
    for f in files:
        img = Image.open(os.path.join(folder, f))
        size.add(img.size)

    print(f"{folder}: {size}")