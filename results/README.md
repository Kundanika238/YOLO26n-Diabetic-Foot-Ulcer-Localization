
# Final YOLO26n Research Results

This folder contains the key experimental results from the final YOLO26n diabetic foot ulcer localization study.

## Final Research Dataset

- Total images: 7,077
- Training images: 5,785
- Validation images: 655
- Test images: 637
- Total annotated DFU regions: 7,755

## Reported Test Performance

| Metric | Result |
|---|---|
| Precision | 78.88% |
| Recall | 62.54% |
| F1-score | 69.77% |
| mAP@0.5 | 70.42% |
| mAP@0.5:0.95 | 35.22% |

These are the recorded final research results. An independent rerun of the final test evaluation has not yet been completed.

## Additional Analyses

### Region-Size Detection Analysis

The study evaluated detection rates for small, medium, and large annotated ulcer regions on the held-out test set.

### Selective High-Resolution Refinement

Selective refinement was evaluated separately on validation images. It increased recall but reduced precision and F1-score while increasing processing time.

The original single-pass YOLO26n model was retained for the final test evaluation.

### Inference Performance

Inference timing was measured separately to assess the computational performance of the trained detector.

## Important Note

The model is intended for research and educational demonstration. It has not been clinically validated and must not be used as a standalone medical diagnostic system.
