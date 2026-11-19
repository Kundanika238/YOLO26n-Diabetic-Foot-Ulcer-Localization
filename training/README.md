
# YOLO26n Training Results

This folder contains the training records of the final
YOLO26n model used for diabetic foot ulcer localization.

## Training Configuration

- Model: YOLO26n (nano)
- Dataset: Screened DFU research dataset
- Training images: 5,785
- Validation images: 655
- Test images: 637
- Epochs: 50
- Input resolution: 512 x 512
- Batch size: 4
- Optimizer: Automatic
- Random seed: 42
- GPU: NVIDIA GeForce GTX 1650 (4 GB)

## Files

- `results.csv`: Recorded training and validation metrics
  across all 50 epochs.

The epoch-wise validation metrics are separate from
the final held-out test evaluation results.
