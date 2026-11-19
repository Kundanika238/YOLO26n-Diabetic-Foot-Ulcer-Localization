
# Final YOLO26n Training Configuration

## Experiment

This document records the training configuration of the final YOLO26n diabetic foot ulcer localization experiment conducted in November 2026.

The detector was initialized using COCO-pretrained YOLO26n weights and fine-tuned for a single diabetic foot ulcer (DFU) detection class.

## Dataset

| Subset | Images | Annotated Regions |
|---|---:|---:|
| Training | 5,785 | 6,404 |
| Validation | 655 | 681 |
| Testing | 637 | 670 |
| Total | 7,077 | 7,755 |

The final dataset was obtained after annotation checks, duplicate screening, and exclusion of selected similarity candidates. The test subset was reserved for final evaluation.

## Model and Training Settings

| Parameter | Recorded Setting |
|---|---|
| Model | YOLO26n |
| Initial weights | COCO-pretrained YOLO26n |
| Training epochs | 50 |
| Input image size | 512 × 512 |
| Batch size | 4 |
| Optimizer selection | Automatic |
| Random seed | 42 |
| Initial learning-rate setting | 0.01 |
| Final learning-rate factor | 0.01 |
| Momentum setting | 0.937 |
| Weight decay | 0.0005 |
| Horizontal flipping | 0.5 |
| Vertical flipping | 0.0 |
| Translation | 0.1 |
| Scaling | 0.5 |
| Mosaic augmentation | Enabled |
| Mosaic closing period | Last 10 epochs |
| MixUp | Disabled |
| Mixed precision | Enabled |
| Training hardware | NVIDIA GeForce GTX 1650 (4 GB) |
| Ultralytics version | 8.4.146 |

The optimizer was selected automatically, so its effective parameters may differ from the recorded initial settings.

## Training Run Provenance

Original local run directory:

`05_Models/yolo26n_research_baseline_50epochs_v2/`

The original training script specifies `resume=False` and initializes from pretrained YOLO26n weights. However, the saved Ultralytics `args.yaml` records a checkpoint-resume path. The exact resume history has not been independently reconstructed.

The repository preserves the final checkpoint and its 50-epoch metrics rather than claiming that the entire training run was uninterrupted.

## Retained Artifacts

- `../models/best.pt` — final trained detector checkpoint.
- `results.csv` — recorded training and validation metrics across 50 epochs.
- `../src/train_yolo26n_research_baseline.py` — original research training script.

The original training script contains local project paths and requires configuration before being executed on another computer.

## Evaluation Scope

The 655-image validation subset was used to monitor training and investigate selective high-resolution refinement.

The 637-image test subset was used to report final single-pass YOLO26n performance. Validation and test results must not be treated as interchangeable.
