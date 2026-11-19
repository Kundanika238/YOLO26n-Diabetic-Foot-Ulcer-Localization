
from pathlib import Path
from collections import defaultdict
import hashlib

DATASET = Path(
    r"D:\Diabetic_Foot_Ulcer_Project\03_YOLO_Dataset\DFU_Research_SCREENED_DRAFT"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

hash_groups = defaultdict(list)

for split in ["train", "val", "test"]:

    image_dir = DATASET / split / "images"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    print(f"Checking {split}: {len(images)} images")

    for image in images:
        digest = hashlib.sha256()

        with image.open("rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024), b""
            ):
                digest.update(chunk)

        hash_groups[digest.hexdigest()].append(
            (split, str(image))
        )

leakage = {
    key: records
    for key, records in hash_groups.items()
    if len({split for split, path in records}) > 1
}

print("\n" + "=" * 55)
print("SCREENED DATASET EXACT LEAKAGE CHECK")
print("=" * 55)

print("Total images checked:",
      sum(len(v) for v in hash_groups.values()))

print("Unique SHA256 hashes:", len(hash_groups))
print("Cross-split exact duplicate groups:", len(leakage))

for records in list(leakage.values())[:5]:
    print("\nDuplicate group:")
    for split, path in records:
        print(split, path)

if len(leakage) == 0:
    print("\nPASS: No exact cross-split duplicates.")
else:
    print("\nREVIEW REQUIRED: Exact leakage detected.")

print("\nNo dataset files were modified.")
