from pathlib import Path
from collections import Counter
import math

DATASET = Path(
    r"D:\Diabetic_Foot_Ulcer_Project\03_YOLO_Dataset\DFU_Research_SCREENED_DRAFT"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_images = 0
total_positive = 0
total_negative = 0
total_boxes = 0
total_errors = 0

for split in ["train", "val", "test"]:

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = list(label_dir.glob("*.txt"))

    image_stems = [p.stem for p in images]
    label_stems = [p.stem for p in labels]

    missing = set(image_stems) - set(label_stems)
    orphan = set(label_stems) - set(image_stems)

    duplicate_image_stems = sum(
        n - 1 for n in Counter(image_stems).values() if n > 1
    )

    positive = 0
    negative = 0
    boxes = 0
    errors = []

    for label_file in labels:

        lines = label_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()

        valid_boxes_in_file = 0

        for line_number, line in enumerate(lines, 1):

            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                errors.append((label_file.name, line_number, "Invalid format"))
                continue

            try:
                numbers = [float(v) for v in values]
                cls, x, y, w, h = numbers

                valid = (
                    all(math.isfinite(v) for v in numbers)
                    and cls == 0
                    and 0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                    and x - w / 2 >= -1e-5
                    and x + w / 2 <= 1 + 1e-5
                    and y - h / 2 >= -1e-5
                    and y + h / 2 <= 1 + 1e-5
                )

                if valid:
                    valid_boxes_in_file += 1
                else:
                    errors.append((label_file.name, line_number, "Invalid values"))

            except ValueError:
                errors.append((label_file.name, line_number, "Non-numeric values"))

        if valid_boxes_in_file > 0:
            positive += 1
        else:
            negative += 1

        boxes += valid_boxes_in_file

    split_errors = len(errors) + len(missing) + len(orphan) + duplicate_image_stems

    print(f"\n{'=' * 45}")
    print(f"SPLIT: {split.upper()}")
    print(f"{'=' * 45}")
    print(f"Images                  : {len(images)}")
    print(f"Annotation files        : {len(labels)}")
    print(f"Positive label files    : {positive}")
    print(f"Negative label files    : {negative}")
    print(f"Valid DFU bounding boxes: {boxes}")
    print(f"Missing labels          : {len(missing)}")
    print(f"Orphan labels           : {len(orphan)}")
    print(f"Duplicate image stems   : {duplicate_image_stems}")
    print(f"Annotation errors       : {len(errors)}")

    for error in errors[:5]:
        print("Example error:", error)

    total_images += len(images)
    total_positive += positive
    total_negative += negative
    total_boxes += boxes
    total_errors += split_errors

print(f"\n{'=' * 45}")
print("FINAL DATASET SUMMARY")
print(f"{'=' * 45}")
print("Total images       :", total_images)
print("Positive labels    :", total_positive)
print("Negative labels    :", total_negative)
print("Valid bounding boxes:", total_boxes)
print("Total reported errors:", total_errors)

if total_errors == 0:
    print("\nBasic annotation audit: PASSED")
else:
    print("\nBasic annotation audit: REVIEW REQUIRED")


