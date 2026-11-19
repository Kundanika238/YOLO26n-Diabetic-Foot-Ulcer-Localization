
from pathlib import Path
from collections import Counter
import csv
import shutil

PROJECT = Path(r"D:\Diabetic_Foot_Ulcer_Project")
AUDIT = PROJECT / "06_Results" / "dataset_audit"

SOURCE = PROJECT / "03_YOLO_Dataset" / "DFU_Research_DRAFT"
OUTPUT = PROJECT / "03_YOLO_Dataset" / "DFU_Research_SCREENED_DRAFT"

LEAKAGE_REPORT = AUDIT / "draft_residual_cross_split_candidates.csv"
SELECTION_REPORT = AUDIT / "draft_representative_selection.csv"

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

# Load residual similarity candidates
leakage = read_csv(LEAKAGE_REPORT)
selection = read_csv(SELECTION_REPORT)

# Hold both groups in every unresolved cross-split pair
hold_groups = set()

for row in leakage:
    hold_groups.add(row["Group_1"])
    hold_groups.add(row["Group_2"])

# Select only representatives copied to the first draft
active = [
    row for row in selection
    if row["Status"] == "COPIED_TO_DRAFT"
]

active_ids = [row["Group_ID"] for row in active]

if len(active) != 7510:
    raise RuntimeError("Unexpected draft representative count")

if len(active_ids) != len(set(active_ids)):
    raise RuntimeError("Duplicate group IDs in selection report")

if not hold_groups.issubset(set(active_ids)):
    raise RuntimeError("A held group is missing from the draft")

if OUTPUT.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT}"
    )

# Validate the complete copy plan before creating files
copy_plan = []
hold_records = []
counts = Counter()

for row in active:
    group_id = row["Group_ID"]
    split = row["Proposed_Split"]

    if split not in ("train", "val", "test"):
        raise RuntimeError(f"Invalid split: {split}")

    if group_id in hold_groups:
        counts[f"{split}_held"] += 1

        hold_records.append({
            "Group_ID": group_id,
            "Split": split,
            "Reason": "Residual cross-split similarity candidate",
            "Status": "HOLD_PENDING_REVIEW"
        })
        continue

    extension = Path(row["Selected_Image"]).suffix.lower()

    source_image = SOURCE / split / "images" / (
        group_id + extension
    )

    source_label = SOURCE / split / "labels" / (
        group_id + ".txt"
    )

    if not source_image.is_file():
        raise FileNotFoundError(source_image)

    if not source_label.is_file():
        raise FileNotFoundError(source_label)

    copy_plan.append((
        group_id, split,
        row["Label_Type"],
        source_image, source_label
    ))

    counts[f"{split}_copied"] += 1
    counts[f"{split}_{row['Label_Type']}"] += 1

if len(copy_plan) != 7077:
    raise RuntimeError("Unexpected screened dataset count")

# Build a separate dataset; original files remain untouched
for split in ["train", "val", "test"]:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

for group_id, split, label_type, source_image, source_label in copy_plan:

    shutil.copy2(
        source_image,
        OUTPUT / split / "images" / source_image.name
    )

    shutil.copy2(
        source_label,
        OUTPUT / split / "labels" / source_label.name
    )

# Save the groups temporarily excluded
hold_report = AUDIT / "screened_residual_hold_groups.csv"

with hold_report.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["Group_ID", "Split", "Reason", "Status"]
    )
    writer.writeheader()
    writer.writerows(hold_records)

print("=" * 60)
print("DFU SCREENED RESEARCH DATASET - DRAFT")
print("=" * 60)

print("\nOriginal draft representatives:", len(active))
print("Groups held for residual similarity:", len(hold_groups))
print("Representatives copied:", len(copy_plan))

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}")
    print("Copied:", counts[f"{split}_copied"])
    print("Positive:", counts[f"{split}_Positive"])
    print("Negative:", counts[f"{split}_Negative"])
    print("Held:", counts[f"{split}_held"])

print("\nScreened dataset location:")
print(OUTPUT)

print("\nHold report:")
print(hold_report)

print("\nIMPORTANT:")
print("This is still a research dataset draft.")
print("Source labels and patient independence are not verified.")
print("No original images, labels, or model weights were changed.")
print("Do not train until final checks are complete.")
