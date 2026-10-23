import os
from PIL import Image

TARGET_SIZE = (256, 256) # Common size for both classes
FOLDERS = ["data/cinematic", "data/normal"]

def normalize_folder(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nProcessing {folder} ({len(files)} images)...")

    fixed = 0
    skipped = 0

    for i, file_name in enumerate(files):
        path = os.path.join(folder, file_name)
        try:
            img = Image.open(path).convert("RGB")

            # Resize the image to the target size
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.BICUBIC)
                img.save(path, quality=95)
                fixed += 1
            else:
                skipped += 1
        
        except Exception as e:
            print(f" Skipping corrupt file {file_name}: {e}")

        if (i + 1) % 2000 == 0:
            print(f" ...{i + 1}/{len(files)} processed")

    print(f"Done: {folder} - resized {fixed}, already corrected {skipped}")


def main():
    print(f"Normalizing all images to {TARGET_SIZE} using BICUBIC resampling")
    print("Warning: This will overwrite the original images. Make a backup if needed.\n")

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            print(f"Folder not found: {folder}")
            continue
        normalize_folder(folder)

    print("\nAll resized to 256x256.")


if __name__ == "__main__":
    main()
