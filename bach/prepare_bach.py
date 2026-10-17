import csv
import random
from pathlib import Path
from PIL import Image

source = Path("/work/nvme/bfqi") / Path.home().name / "Photos"
output = Path.cwd() / "patches"
manifest_path = Path.cwd() / "manifest.csv"

rng = random.Random(42)
records = []

with open(source / "microscopy_ground_truth.csv") as f:
    labels = list(csv.reader(f))

# Split independently within each diagnosis category.
splits = {}

for label in ["Normal", "Benign", "InSitu", "Invasive"]:
    names = [name for name, cls in labels if cls == label]
    assert len(names) == 100
    rng.shuffle(names)

    for i, name in enumerate(names):
        split = "train" if i < 80 else "val" if i < 90 else "test"
        splits[name] = split

# Select 12 spatially distributed patches per original image.
for name, label in labels:
    split = splits[name]
    path = source / label / name

    with Image.open(path) as original:
        img = original.convert("RGB")
        w, h = img.size

        for row in range(3):
            for col in range(4):
                x = round((w - 224) * col / 3)
                y = round((h - 224) * row / 2)

                patch = img.crop((x, y, x + 224, y + 224))

                folder = output / split / label / Path(name).stem
                folder.mkdir(parents=True, exist_ok=True)

                patch_path = folder / f"r{row}_c{col}.png"
                patch.save(patch_path)

                records.append([
                    str(patch_path), name, label, split
                ])

with open(manifest_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "original_image", "label", "split"])
    writer.writerows(records)

print("Original images:", len(labels))
print("Patches:", len(records))
print("Manifest:", manifest_path)
print("Done!")
