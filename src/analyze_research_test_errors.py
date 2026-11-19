
from pathlib import Path
from ultralytics import YOLO
import csv

ROOT = Path(r"D:\Diabetic_Foot_Ulcer_Project")

DATA = ROOT / "03_YOLO_Dataset" / "DFU_Research_SCREENED_DRAFT"
WEIGHTS = ROOT / "05_Models" / "yolo26n_research_baseline_50epochs_v2" / "weights" / "best.pt"
OUTPUT = ROOT / "06_Results" / "research_test_error_analysis.csv"

CONF = 0.25
MATCH_IOU = 0.50

if not WEIGHTS.is_file():
    raise FileNotFoundError(WEIGHTS)

if OUTPUT.exists():
    raise FileExistsError(f"Report already exists: {OUTPUT}")


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0


def read_ground_truth(image_path, width, height):
    label = DATA / "test" / "labels" / (image_path.stem + ".txt")

    if not label.is_file():
        raise FileNotFoundError(label)

    boxes = []

    for line in label.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue

        cls, cx, cy, bw, bh = map(float, line.split())

        if int(cls) != 0:
            raise ValueError(f"Unexpected class in {label}")

        boxes.append([
            (cx - bw/2) * width,
            (cy - bh/2) * height,
            (cx + bw/2) * width,
            (cy + bh/2) * height
        ])

    return boxes


print("=" * 60)
print("DFU TEST DATASET - DETECTION ERROR ANALYSIS")
print("=" * 60)

model = YOLO(str(WEIGHTS))
rows = []

predictions = model.predict(
    source=str(DATA / "test" / "images"),
    imgsz=512,
    conf=CONF,
    iou=0.70,
    device=0,
    stream=True,
    save=False,
    verbose=False
)

for result in predictions:
    image_path = Path(result.path)
    height, width = result.orig_shape

    gt_boxes = read_ground_truth(image_path, width, height)

    pred_boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    ordered_predictions = sorted(
        zip(pred_boxes, confidences),
        key=lambda item: item[1],
        reverse=True
    )

    unmatched_gt = set(range(len(gt_boxes)))

    tp = 0
    fp = 0

    for pred_box, confidence in ordered_predictions:
        best_index = None
        best_iou = 0.0

        for index in unmatched_gt:
            overlap = box_iou(pred_box, gt_boxes[index])

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= MATCH_IOU:
            tp += 1
            unmatched_gt.remove(best_index)
        else:
            fp += 1

    fn = len(unmatched_gt)

    rows.append({
        "Image": image_path.name,
        "GT_Boxes": len(gt_boxes),
        "Predicted_Boxes": len(pred_boxes),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Error_Total": fp + fn
    })

if len(rows) != 637:
    raise RuntimeError(f"Expected 637 images, processed {len(rows)}")

fields = [
    "Image", "GT_Boxes", "Predicted_Boxes",
    "TP", "FP", "FN", "Error_Total"
]

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

tp = sum(r["TP"] for r in rows)
fp = sum(r["FP"] for r in rows)
fn = sum(r["FN"] for r in rows)

images_with_errors = sum(r["Error_Total"] > 0 for r in rows)

print("\nTEST IMAGES:", len(rows))
print("TRUE POSITIVES:", tp)
print("FALSE POSITIVES:", fp)
print("FALSE NEGATIVES:", fn)
print("IMAGES WITH DETECTION ERRORS:", images_with_errors)

print("\nTOP 10 IMAGES FOR VISUAL REVIEW")

worst = sorted(
    rows,
    key=lambda r: (-r["Error_Total"], -r["FN"], r["Image"])
)

for row in worst[:10]:
    print(
        f"{row['Image']} | "
        f"TP={row['TP']} FP={row['FP']} FN={row['FN']}"
    )

print("\nREPORT SAVED:")
print(OUTPUT)

print("\nNOTE:")
print("Matching uses confidence 0.25 and IoU 0.50.")
print("These descriptive counts may differ from Ultralytics")
print("summary metrics and its confusion matrix.")
print("No training or dataset files were modified.")
