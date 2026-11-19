from pathlib import Path
from collections import defaultdict
import csv
import time

import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(r"D:\Diabetic_Foot_Ulcer_Project")

MODEL_PATH = (
    ROOT / "05_Models"
    / "yolo26n_research_baseline_50epochs_v2"
    / "weights" / "best.pt"
)

DATA = (
    ROOT / "03_YOLO_Dataset"
    / "DFU_Research_SCREENED_DRAFT"
)

OUTPUT = (
    ROOT / "06_Results"
    / "FINAL_Research_Results"
    / "Selective_Refinement_Validation"
)

CONF = 0.25
TRIGGER_CONF = 0.60
MATCH_IOU = 0.50
MERGE_IOU = 0.50
DEVICE = 0

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable.")

images = sorted(
    p for p in (DATA / "val" / "images").iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(images) != 655:
    raise RuntimeError(
        f"Expected 655 validation images, found {len(images)}"
    )

OUTPUT.mkdir(parents=True)

model = YOLO(str(MODEL_PATH))


def sync_gpu():
    torch.cuda.synchronize()


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def category(area):
    if area < 0.01:
        return "Small"
    if area < 0.05:
        return "Medium"
    return "Large"


def read_labels(path, width, height):
    label = DATA / "val" / "labels" / f"{path.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(label)

    ground_truth = []

    for line in label.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        if not line.strip():
            continue

        parts = list(map(float, line.split()))

        if len(parts) != 5 or int(parts[0]) != 0:
            raise ValueError(f"Invalid label: {label}")

        _, cx, cy, bw, bh = parts

        ground_truth.append({
            "box": [
                (cx-bw/2)*width,
                (cy-bh/2)*height,
                (cx+bw/2)*width,
                (cy+bh/2)*height
            ],
            "size": category(bw * bh)
        })

    return ground_truth


def predict_array(image):
    result = model.predict(
        source=image,
        imgsz=512,
        conf=CONF,
        iou=0.70,
        device=DEVICE,
        verbose=False
    )[0]

    detections = []

    for box in result.boxes:
        if int(box.cls[0]) != 0:
            continue

        detections.append({
            "box": box.xyxy[0].cpu().tolist(),
            "conf": float(box.conf[0])
        })

    return detections


def make_tiles(image):
    h, w = image.shape[:2]

    tile_w = min(w, max(32, round(w * 0.60)))
    tile_h = min(h, max(32, round(h * 0.60)))

    positions = {
        (0, 0),
        (w-tile_w, 0),
        (0, h-tile_h),
        (w-tile_w, h-tile_h)
    }

    for x, y in sorted(positions):
        crop = image[
            y:y+tile_h,
            x:x+tile_w
        ]

        yield x, y, crop


def refine(image, baseline):
    # Trigger uses predictions only, never ground truth.
    maximum_conf = max(
        (d["conf"] for d in baseline),
        default=0.0
    )

    triggered = maximum_conf < TRIGGER_CONF

    if not triggered:
        return list(baseline), False

    candidates = []

    for x, y, crop in make_tiles(image):
        for detection in predict_array(crop):

            box = detection["box"]

            candidates.append({
                "box": [
                    box[0] + x,
                    box[1] + y,
                    box[2] + x,
                    box[3] + y
                ],
                "conf": detection["conf"]
            })

    # Preserve original predictions first.
    combined = list(baseline)

    # Add only non-overlapping extra detections.
    candidates.sort(
        key=lambda d: d["conf"],
        reverse=True
    )

    for candidate in candidates:
        duplicate = any(
            box_iou(candidate["box"], d["box"])
            >= MERGE_IOU
            for d in combined
        )

        if not duplicate:
            combined.append(candidate)

    return combined, True


def evaluate(predictions, ground_truth):
    unmatched = set(range(len(ground_truth)))
    detected = set()
    fp = 0

    predictions = sorted(
        predictions,
        key=lambda d: d["conf"],
        reverse=True
    )

    for pred in predictions:
        best_index = None
        best_iou = 0.0

        for index in unmatched:
            overlap = box_iou(
                pred["box"],
                ground_truth[index]["box"]
            )

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= MATCH_IOU:
            detected.add(best_index)
            unmatched.remove(best_index)
        else:
            fp += 1

    stats = defaultdict(
        lambda: {"GT": 0, "TP": 0, "FN": 0}
    )

    for index, gt in enumerate(ground_truth):
        size = gt["size"]
        stats[size]["GT"] += 1

        if index in detected:
            stats[size]["TP"] += 1
        else:
            stats[size]["FN"] += 1

    return stats, len(detected), fp, len(unmatched)


def empty_stats():
    return {
        "GT": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "Time_ms": [],
        "Sizes": defaultdict(
            lambda: {"GT": 0, "TP": 0, "FN": 0}
        )
    }


totals = {
    "Baseline": empty_stats(),
    "Selective": empty_stats()
}

detail_rows = []
triggered_count = 0

print("=" * 65)
print("YOLO26n SELECTIVE HIGH-RESOLUTION VALIDATION")
print("=" * 65)

print("Validation images:", len(images))
print("Confidence threshold:", CONF)
print("Refinement trigger: max confidence <", TRIGGER_CONF)
print("Matching IoU:", MATCH_IOU)
print("No training will be performed.")

# Warm up GPU.
warmup = cv2.imread(str(images[0]))

for _ in range(5):
    predict_array(warmup)

sync_gpu()

for number, path in enumerate(images, start=1):

    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")

    h, w = image.shape[:2]
    ground_truth = read_labels(path, w, h)

    sync_gpu()
    start = time.perf_counter()

    baseline = predict_array(image)

    sync_gpu()
    baseline_ms = (
        time.perf_counter() - start
    ) * 1000

    refined, triggered = refine(image, baseline)

    sync_gpu()
    selective_ms = (
        time.perf_counter() - start
    ) * 1000

    if triggered:
        triggered_count += 1

    for method, detections, elapsed in [
        ("Baseline", baseline, baseline_ms),
        ("Selective", refined, selective_ms)
    ]:

        sizes, tp, fp, fn = evaluate(
            detections, ground_truth
        )

        result = totals[method]

        result["GT"] += len(ground_truth)
        result["TP"] += tp
        result["FP"] += fp
        result["FN"] += fn
        result["Time_ms"].append(elapsed)

        for size, values in sizes.items():
            for key in ["GT", "TP", "FN"]:
                result["Sizes"][size][key] += values[key]

        detail_rows.append({
            "Image": path.name,
            "Method": method,
            "Refinement_Triggered": int(triggered),
            "GT": len(ground_truth),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Time_ms": round(elapsed, 3)
        })

    if number % 50 == 0 or number == len(images):
        print(f"Processed {number}/{len(images)}")


# Validate both methods evaluated the same annotations.
for method in totals:
    if totals[method]["GT"] != 681:
        raise RuntimeError(
            f"{method}: incorrect GT total"
        )


summary_rows = []
size_rows = []

print("\n========== OVERALL RESULTS ==========")

for method in ["Baseline", "Selective"]:
    s = totals[method]

    precision = (
        s["TP"] / (s["TP"] + s["FP"])
        if s["TP"] + s["FP"] else 0
    )

    recall = (
        s["TP"] / (s["TP"] + s["FN"])
        if s["TP"] + s["FN"] else 0
    )

    f1 = (
        2*precision*recall/(precision+recall)
        if precision+recall else 0
    )

    avg_ms = sum(s["Time_ms"]) / len(s["Time_ms"])

    summary_rows.append({
        "Method": method,
        "TP": s["TP"],
        "FP": s["FP"],
        "FN": s["FN"],
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Average_ms": round(avg_ms, 3)
    })

    print(
        f"{method}: "
        f"TP={s['TP']} FP={s['FP']} FN={s['FN']} "
        f"P={precision:.4f} R={recall:.4f} "
        f"F1={f1:.4f} "
        f"Avg={avg_ms:.2f} ms"
    )

    for size in ["Small", "Medium", "Large"]:
        v = s["Sizes"][size]

        rate = (
            100*v["TP"]/v["GT"]
            if v["GT"] else 0
        )

        size_rows.append({
            "Method": method,
            "Size": size,
            "GT": v["GT"],
            "Detected": v["TP"],
            "Missed": v["FN"],
            "Detection_Rate_Percent": round(rate, 2)
        })

print("\nImages receiving refinement:", triggered_count)

print("\n========== SIZE-WISE RESULTS ==========")

for row in size_rows:
    print(
        f'{row["Method"]} | {row["Size"]}: '
        f'{row["Detected"]}/{row["GT"]} '
        f'({row["Detection_Rate_Percent"]:.2f}%)'
    )


def save_csv(filename, records):
    with (OUTPUT / filename).open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


save_csv("comparison_summary.csv", summary_rows)
save_csv("size_comparison.csv", size_rows)
save_csv("image_details.csv", detail_rows)


# Create comparison graph.
sizes = ["Small", "Medium", "Large"]

baseline_rates = [
    next(
        r["Detection_Rate_Percent"]
        for r in size_rows
        if r["Method"] == "Baseline"
        and r["Size"] == size
    )
    for size in sizes
]

selective_rates = [
    next(
        r["Detection_Rate_Percent"]
        for r in size_rows
        if r["Method"] == "Selective"
        and r["Size"] == size
    )
    for size in sizes
]

positions = list(range(3))

plt.figure(figsize=(10, 6))

plt.bar(
    [x - 0.2 for x in positions],
    baseline_rates,
    width=0.4,
    label="Original YOLO26n"
)

plt.bar(
    [x + 0.2 for x in positions],
    selective_rates,
    width=0.4,
    label="Selective Refinement"
)

plt.xticks(positions, sizes)
plt.xlabel("Lesion Size")
plt.ylabel("Detection Rate (%)")
plt.title("Validation: Baseline vs Selective Refinement")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "size_comparison.png",
    dpi=300
)

plt.close()

print("\nResults saved:")
print(OUTPUT)

print("\nIMPORTANT:")
print("This experiment uses validation images only.")
print("Reported metrics are fixed-threshold metrics, not mAP.")
print("The strategy is experimental, not a clinically validated method.")
print("No model weights or dataset annotations were modified.")
