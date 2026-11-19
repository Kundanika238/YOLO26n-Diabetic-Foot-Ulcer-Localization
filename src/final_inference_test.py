from pathlib import Path
import csv
import time
import statistics
import sys

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(r"D:\Diabetic_Foot_Ulcer_Project")

MODEL = (
    ROOT / "05_Models"
    / "yolo26n_research_baseline_50epochs_v2"
    / "weights" / "best.pt"
)

IMAGES = (
    ROOT / "03_YOLO_Dataset"
    / "DFU_Research_SCREENED_DRAFT"
    / "test" / "images"
)

OUTPUT = (
    ROOT / "06_Results"
    / "FINAL_Research_Results"
    / "Inference_Performance"
)

if not MODEL.is_file():
    raise FileNotFoundError(MODEL)

if not IMAGES.is_dir():
    raise FileNotFoundError(IMAGES)

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable.")

images = sorted(
    p for p in IMAGES.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if len(images) != 637:
    raise RuntimeError(
        f"Expected 637 test images, found {len(images)}"
    )

OUTPUT.mkdir(parents=True)

print("=" * 60)
print("YOLO26n FINAL INFERENCE PERFORMANCE TEST")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL.name)
print("Model size (MB):", round(MODEL.stat().st_size / 1048576, 3))
print("Total available test images:", len(images))

model = YOLO(str(MODEL))

# Choose a previously reviewed successful example.
sample = next(
    (p for p in images if p.stem == "G01352"),
    images[0]
)

result = model.predict(
    source=str(sample),
    imgsz=512,
    conf=0.25,
    iou=0.70,
    device=0,
    verbose=False
)[0]

prediction_file = OUTPUT / "single_image_prediction.jpg"
result.save(filename=str(prediction_file))

print("\nSINGLE IMAGE PREDICTION")
print("Image:", sample.name)
print("Detections:", len(result.boxes))
print("Saved:", prediction_file)

# Warm up GPU before measuring inference.
for _ in range(5):
    model.predict(
        source=str(sample),
        imgsz=512,
        conf=0.25,
        iou=0.70,
        device=0,
        verbose=False
    )

torch.cuda.synchronize()

# Select 30 images spread across the test dataset.
indices = [
    round(i * (len(images) - 1) / 29)
    for i in range(30)
]

benchmark_images = [images[i] for i in indices]

records = []

print("\nBENCHMARKING 30 TEST IMAGES")

for index, image in enumerate(benchmark_images, start=1):

    torch.cuda.synchronize()
    start = time.perf_counter()

    prediction = model.predict(
        source=str(image),
        imgsz=512,
        conf=0.25,
        iou=0.70,
        device=0,
        verbose=False
    )[0]

    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000

    records.append({
        "Image": image.name,
        "Detections": len(prediction.boxes),
        "End_to_End_ms": round(elapsed_ms, 3),
        "Preprocessing_ms": round(
            prediction.speed.get("preprocess", 0), 3
        ),
        "Inference_ms": round(
            prediction.speed.get("inference", 0), 3
        ),
        "Postprocessing_ms": round(
            prediction.speed.get("postprocess", 0), 3
        ),
    })

    print(
        f"{index}/30: {image.name} "
        f"| End-to-end: {elapsed_ms:.2f} ms"
    )

# Save CSV.
csv_file = OUTPUT / "inference_performance.csv"

with csv_file.open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)

times = [r["End_to_End_ms"] for r in records]
inference_times = [r["Inference_ms"] for r in records]

average_ms = statistics.mean(times)
median_ms = statistics.median(times)
average_inference = statistics.mean(inference_times)

fps = 1000 / average_ms if average_ms > 0 else 0

summary = f"""
YOLO26n FINAL INFERENCE PERFORMANCE REPORT

Model: YOLO26n
Checkpoint: best.pt
Model size (MB): {MODEL.stat().st_size / 1048576:.3f}

Python: {sys.version.split()[0]}
PyTorch: {torch.__version__}
Ultralytics: {ultralytics.__version__}

GPU: {torch.cuda.get_device_name(0)}
Image size: 512
Confidence threshold: 0.25
NMS IoU threshold: 0.70

Test images benchmarked: {len(records)}

Average end-to-end time (ms): {average_ms:.3f}
Median end-to-end time (ms): {median_ms:.3f}
Average model inference time (ms): {average_inference:.3f}
Approximate sequential throughput (images/sec): {fps:.2f}

NOTES:
- The benchmark uses 30 selected test images.
- The GPU was warmed up before measurement.
- End-to-end timing includes image loading and prediction processing.
- The throughput is an approximate single-image sequential estimate.
- This test does not modify model weights or dataset annotations.
"""

summary_file = OUTPUT / "inference_summary.txt"
summary_file.write_text(summary, encoding="utf-8")

print("\n" + "=" * 60)
print("FINAL INFERENCE RESULTS")
print("=" * 60)

print(f"Average end-to-end time: {average_ms:.3f} ms")
print(f"Median end-to-end time: {median_ms:.3f} ms")
print(f"Average model inference: {average_inference:.3f} ms")
print(f"Approximate throughput: {fps:.2f} images/sec")

print("\nResults saved:")
print(OUTPUT)

print("\nNo training or dataset files were modified.")
