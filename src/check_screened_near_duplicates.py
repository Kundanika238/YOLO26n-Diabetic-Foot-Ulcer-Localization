
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
import numpy as np
import cv2
import hashlib
import csv

PROJECT = Path(r"D:\Diabetic_Foot_Ulcer_Project")

DATASET = (
    PROJECT / "03_YOLO_Dataset" / "DFU_Research_SCREENED_DRAFT"
)

OUTPUT = PROJECT / "06_Results" / "FINAL_Research_Results" / "Screened_NearDuplicate_Audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lower distances mean more similar images.
# This is a candidate-search threshold, not proof of duplication.
MAX_DISTANCE = 6


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def perceptual_hash(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("L")
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(img, dtype=np.float32)

    transform = cv2.dct(pixels)
    frequencies = transform[:8, :8].flatten()[1:]
    median = np.median(frequencies)

    value = 0
    for bit in frequencies > median:
        value = (value << 1) | int(bit)

    return value


def distance(a, b):
    return (a ^ b).bit_count()


class BKNode:
    def __init__(self, value, index):
        self.value = value
        self.indices = [index]
        self.children = {}


class BKTree:
    def __init__(self):
        self.root = None

    def insert(self, value, index):
        if self.root is None:
            self.root = BKNode(value, index)
            return

        node = self.root

        while True:
            d = distance(value, node.value)

            if d == 0:
                node.indices.append(index)
                return

            if d not in node.children:
                node.children[d] = BKNode(value, index)
                return

            node = node.children[d]

    def query(self, value, radius):
        if self.root is None:
            return []

        matches = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            d = distance(value, node.value)

            if d <= radius:
                for index in node.indices:
                    matches.append((index, d))

            low = d - radius
            high = d + radius

            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)

        return matches


records = []
failed = []
tree = BKTree()
counts = Counter()

cross_path = OUTPUT / "near_duplicates_cross_split.csv"
within_path = OUTPUT / "near_duplicates_within_split.csv"

fields = [
    "Split_1", "Image_1",
    "Split_2", "Image_2",
    "pHash_Distance", "Review_Priority"
]

cross_examples = []

print("=" * 55)
print("DFU PERCEPTUAL NEAR-DUPLICATE AUDIT")
print("=" * 55)

with (
    cross_path.open("w", newline="", encoding="utf-8") as cf,
    within_path.open("w", newline="", encoding="utf-8") as wf
):
    cross_writer = csv.DictWriter(cf, fieldnames=fields)
    within_writer = csv.DictWriter(wf, fieldnames=fields)

    cross_writer.writeheader()
    within_writer.writeheader()

    for split in ["train", "val", "test"]:
        folder = DATASET / split / "images"

        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        print(f"\nProcessing {split}: {len(images)} images")

        for path in images:
            try:
                phash = perceptual_hash(path)
                exact_hash = sha256_file(path)
            except Exception as exc:
                failed.append((str(path), str(exc)))
                continue

            current = {
                "split": split,
                "path": str(path),
                "sha256": exact_hash
            }

            matches = tree.query(phash, MAX_DISTANCE)

            for old_index, d in matches:
                previous = records[old_index]

                # Exact duplicates were already audited.
                if previous["sha256"] == exact_hash:
                    continue

                cross_split = previous["split"] != split
                priority = (
                    "Strong candidate" if d <= 3
                    else "Possible near-duplicate"
                )

                row = {
                    "Split_1": previous["split"],
                    "Image_1": previous["path"],
                    "Split_2": split,
                    "Image_2": str(path),
                    "pHash_Distance": d,
                    "Review_Priority": priority
                }

                if cross_split:
                    cross_writer.writerow(row)
                    counts["cross"] += 1

                    pair = "-".join(
                        sorted([previous["split"], split])
                    )
                    counts[pair] += 1

                    if len(cross_examples) < 15:
                        cross_examples.append(row)
                else:
                    within_writer.writerow(row)
                    counts["within"] += 1

                counts[priority] += 1

            index = len(records)
            records.append(current)
            tree.insert(phash, index)


print("\n" + "=" * 55)
print("NEAR-DUPLICATE AUDIT RESULTS")
print("=" * 55)

print("Images successfully checked:", len(records))
print("Unreadable/failed images:", len(failed))

print("\nNon-identical candidate image pairs:")
print("Cross-split:", counts["cross"])
print("Within-split:", counts["within"])

print("\nCross-split candidate pairs:")
print("Train-Val :", counts["train-val"])
print("Train-Test:", counts["test-train"])
print("Val-Test  :", counts["test-val"])

print("\nPriority distribution:")
print("Strong candidates:", counts["Strong candidate"])
print("Possible candidates:", counts["Possible near-duplicate"])

print("\nEXAMPLE CROSS-SPLIT CANDIDATES")

for row in cross_examples:
    print(
        f"{row['Split_1']}/{Path(row['Image_1']).name}"
        f" <-> {row['Split_2']}/{Path(row['Image_2']).name}"
        f" | Distance: {row['pHash_Distance']}"
    )

print("\nReports saved:")
print(cross_path)
print(within_path)

if failed:
    print("\nFirst unreadable images:")
    for item in failed[:10]:
        print(item)

print("\nIMPORTANT:")
print("These are similarity candidates, not confirmed duplicates.")
print("Manual review is necessary.")
print("Original dataset was not modified.")

