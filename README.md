
# YOLO26n for Diabetic Foot Ulcer Localization

A deep learning research project for diabetic foot ulcer (DFU) localization using YOLO26n, with region-size performance analysis, selective high-resolution refinement, and a Flask-based web application named DFUVision.

## Project Overview

The project uses a COCO-pretrained YOLO26n detector fine-tuned for single-class diabetic foot ulcer localization. The model predicts bounding boxes and confidence scores around suspected ulcer regions in foot photographs.

## Dataset

The final screened research dataset consists of 7,077 images and 7,755 annotated DFU regions from eight sources.

| Subset | Images | Bounding Boxes |
|---|---:|---:|
| Training | 5,785 | 6,404 |
| Validation | 655 | 681 |
| Testing | 637 | 670 |
| Total | 7,077 | 7,755 |

Dataset images are not redistributed in this repository.

## Training Configuration

- Model: YOLO26n
- Epochs: 50
- Input resolution: 512 × 512
- Batch size: 4
- Optimizer: Automatic
- Random seed: 42
- GPU: NVIDIA GeForce GTX 1650 (4 GB)
- Framework: PyTorch and Ultralytics

## Reported Test Results

| Metric | Result |
|---|---:|
| Precision | 78.88% |
| Recall | 62.54% |
| F1-score | 69.77% |
| mAP@0.5 | 70.42% |
| mAP@0.5:0.95 | 35.22% |

These are the recorded final test results. An independent rerun has not been completed.

## Region-Size Performance

At confidence ≥ 0.25 and matching IoU ≥ 0.50:

| Region Size | Matched / Total | Detection Rate |
|---|---:|---:|
| Small (<1%) | 28 / 76 | 36.84% |
| Medium (1% to <5%) | 205 / 332 | 61.75% |
| Large (≥5%) | 191 / 262 | 72.90% |

These are fixed-threshold matched-region detection rates, not COCO size-specific AP.

## Selective High-Resolution Refinement

Selective refinement was tested on the validation dataset. Images with a maximum detection confidence below 0.60 were processed using four overlapping crops.

The refinement increased recall but also increased false positives and processing time, reducing F1-score. The original single-pass model was retained for final test evaluation.

## DFUVision Web Application

DFUVision is a Flask-based demonstration interface that supports:

- Foot image upload and preview
- Predicted ulcer bounding boxes
- Confidence scores
- Detection counts
- Annotated image downloads
- Optional region inspection

## Repository Structure

- `models/` — final trained YOLO26n checkpoint
- `training/` — 50-epoch training history
- `results/` — experimental result summaries
- `src/` — research implementation scripts
- `webapp/` — DFUVision Flask application

## Limitations

Patient-independent dataset splitting was not verified. Potential undetected near-duplicates and annotation-quality limitations may affect generalizability.

Some research scripts require external dataset files and local path configuration. The repository does not currently provide a fully automated reproduction of every experiment.

## Disclaimer

This project is for research and educational purposes only. It has not been clinically validated and must not be used as a standalone medical diagnostic tool.
