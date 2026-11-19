from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(r"D:\Diabetic_Foot_Ulcer_Project")

MODEL = (
    ROOT / "05_Models"
    / "yolo26n_research_baseline_50epochs_v2"
    / "weights" / "best.pt"
)

DATASET = (
    ROOT / "03_YOLO_Dataset"
    / "DFU_Research_SCREENED_DRAFT"
)

OUTPUT = (
    ROOT / "06_Results"
    / "FINAL_Research_Results"
    / "Lesion_Size_Analysis"
)

CONF = 0.25
MATCH_IOU = 0.50

if not MODEL.is_file():
    raise FileNotFoundError(MODEL)

if OUTPUT.exists():
    raise FileExistsError(
        f"Output folder already exists: {OUTPUT}"
    )

OUTPUT.mkdir(parents=True)

def size_category(area_fraction):

    if area_fraction < 0.01:
        return "Small"

    if area_fraction < 0.05:
        return "Medium"

    return "Large"


def calculate_iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = (
        max(0, a[2] - a[0]) *
        max(0, a[3] - a[1])
    )

    area_b = (
        max(0, b[2] - b[0]) *
        max(0, b[3] - b[1])
    )

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


print("=" * 65)
print("YOLO26n - DFU LESION SIZE ANALYSIS")
print("=" * 65)

model = YOLO(str(MODEL))

stats = defaultdict(
    lambda: {"GT": 0, "TP": 0, "FN": 0}
)

annotation_rows = []

total_images = 0
total_fp = 0

results = model.predict(
    source=str(DATASET / "test" / "images"),
    imgsz=512,
    conf=CONF,
    iou=0.70,
    device=0,
    stream=True,
    save=False,
    verbose=False
)

for result in results:

    total_images += 1

    image_path = Path(result.path)
    height, width = result.orig_shape

    label_path = (
        DATASET / "test" / "labels"
        / f"{image_path.stem}.txt"
    )

    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    ground_truth = []

    for line in label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        if not line.strip():
            continue

        cls, cx, cy, bw, bh = map(
            float, line.split()
        )

        if int(cls) != 0:
            raise ValueError(
                f"Unexpected class in {label_path}"
            )

        category = size_category(bw * bh)

        ground_truth.append({
            "box": [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height
            ],
            "category": category,
            "area_pct": bw * bh * 100
        })

    predictions = []

    for box in result.boxes:

        predictions.append({
            "box": box.xyxy[0].cpu().tolist(),
            "confidence": float(box.conf[0])
        })

    predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    unmatched_gt = set(range(len(ground_truth)))
    matched = {}
    fp = 0

    for pred in predictions:

        best_index = None
        best_iou = 0.0

        for index in unmatched_gt:

            overlap = calculate_iou(
                pred["box"],
                ground_truth[index]["box"]
            )

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if (
            best_index is not None
            and best_iou >= MATCH_IOU
        ):

            matched[best_index] = {
                "confidence": pred["confidence"],
                "iou": best_iou
            }

            unmatched_gt.remove(best_index)

        else:
            fp += 1

    total_fp += fp

    for index, gt in enumerate(ground_truth):

        category = gt["category"]
        detected = index in matched

        stats[category]["GT"] += 1

        if detected:
            stats[category]["TP"] += 1
        else:
            stats[category]["FN"] += 1

        match = matched.get(index)

        annotation_rows.append({
            "Image": image_path.name,
            "Annotation_Index": index + 1,
            "Area_Percent": round(
                gt["area_pct"], 4
            ),
            "Size_Category": category,
            "Detected": int(detected),
            "Matched_Confidence": (
                round(match["confidence"], 4)
                if match else ""
            ),
            "Matched_IoU": (
                round(match["iou"], 4)
                if match else ""
            )
        })


if total_images != 637:
    raise RuntimeError(
        f"Expected 637 images, found {total_images}"
    )

total_gt = sum(
    stats[c]["GT"]
    for c in ["Small", "Medium", "Large"]
)

total_tp = sum(
    stats[c]["TP"]
    for c in ["Small", "Medium", "Large"]
)

total_fn = sum(
    stats[c]["FN"]
    for c in ["Small", "Medium", "Large"]
)

if total_gt != 670:
    raise RuntimeError(
        f"Expected 670 annotations, found {total_gt}"
    )

summary = []

for category in ["Small", "Medium", "Large"]:

    counts = stats[category]

    rate = (
        counts["TP"] / counts["GT"] * 100
        if counts["GT"] > 0 else None
    )

    summary.append({
        "Size": category,
        "Annotations": counts["GT"],
        "Detected": counts["TP"],
        "Missed": counts["FN"],
        "Detection_Rate_Percent": (
            round(rate, 2)
            if rate is not None else "N/A"
        )
    })


# Save summary CSV
summary_file = OUTPUT / "lesion_size_summary.csv"

with summary_file.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )

    writer.writeheader()
    writer.writerows(summary)


# Save every annotation's matching result
detail_file = OUTPUT / "lesion_size_details.csv"

with detail_file.open(
    "w", newline="", encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f, fieldnames=list(annotation_rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(annotation_rows)


# Create graph
categories = [row["Size"] for row in summary]

rates = [
    (
        row["Detection_Rate_Percent"]
        if isinstance(
            row["Detection_Rate_Percent"], (int, float)
        )
        else 0
    )
    for row in summary
]

plt.figure(figsize=(10, 6))

bars = plt.bar(categories, rates)

for bar, row in zip(bars, summary):

    if row["Annotations"] == 0:
        label = "N/A - no annotations"
    else:
        label = (
            f'{row["Detection_Rate_Percent"]:.2f}% '
            f'({row["Detected"]}/{row["Annotations"]})'
        )

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        label,
        ha="center",
        fontsize=11
    )

plt.title("YOLO26n Detection Rate by Ulcer Size")
plt.xlabel("Ulcer Size Category")
plt.ylabel("Detection Rate (%)")
plt.ylim(0, 110)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

graph_file = OUTPUT / "lesion_size_detection_rate.png"

plt.savefig(
    graph_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Save text report
report_file = OUTPUT / "lesion_size_report.txt"

report_lines = [
    "YOLO26n LESION SIZE DETECTION ANALYSIS",
    "",
    "Test images: 637",
    f"Annotated boxes: {total_gt}",
    f"Matched boxes: {total_tp}",
    f"Missed boxes: {total_fn}",
    f"False positive predictions: {total_fp}",
    "",
    "Size definitions based on image area:",
    "Small: <1%",
    "Medium: >=1% and <5%",
    "Large: >=5%",
    "",
    "Detection confidence threshold: 0.25",
    "Matching IoU threshold: 0.50",
    "",
]

for row in summary:
    report_lines.append(
        f'{row["Size"]}: '
        f'{row["Detected"]}/{row["Annotations"]} detected, '
        f'rate = {row["Detection_Rate_Percent"]}%'
    )

report_lines.extend([
    "",
    "These are exploratory image-relative size categories.",
    "Detection rate is matched boxes / annotated boxes.",
    "It is not COCO AP or mAP.",
    "False positives are not assigned to lesion-size groups.",
    "Patient independence and clinical labels are unverified."
])

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)


print("\n========== FINAL RESULTS ==========")

for row in summary:
    print(
        f'{row["Size"]}: '
        f'GT={row["Annotations"]}, '
        f'TP={row["Detected"]}, '
        f'FN={row["Missed"]}, '
        f'Detection rate={row["Detection_Rate_Percent"]}%'
    )

print("\nTOTAL ANNOTATIONS:", total_gt)
print("TOTAL DETECTED:", total_tp)
print("TOTAL MISSED:", total_fn)
print("TOTAL FALSE POSITIVES:", total_fp)

if (total_tp, total_fn, total_fp) == (424, 246, 185):
    print("\nCHECK: Matches previous fixed-threshold error analysis.")
else:
    print("\nCHECK: Counts differ from previous error analysis.")
    print("Review matching settings before reporting results.")

print("\nRESULTS SAVED:")
print(OUTPUT)

print("\nNo model training or dataset modifications performed.")
