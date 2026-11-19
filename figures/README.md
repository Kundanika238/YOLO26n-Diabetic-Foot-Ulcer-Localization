# Final Research Figures

This folder contains selected visualizations from the final YOLO26n diabetic foot ulcer localization experiment.

## Included Figures

- `04_Precision_Recall_F1.png` — training and validation performance visualization.
- `05_mAP.png` — mean average precision across training epochs.
- `BoxPR_curve.png` — final test precision–recall curve.
- `BoxF1_curve.png` — final test F1-score curve.
- `lesion_size_detection_rate.png` — detection performance by annotated ulcer-region size.
- `size_comparison.png` — validation-only selective refinement analysis.

## Evaluation Notes

Training and validation curves are distinct from final test results.

Region-size detection rates use a confidence threshold of 0.25 and matching IoU of 0.50.

Selective refinement was evaluated only on the validation dataset. Final test evaluation retained the original single-pass YOLO26n detector.

These visualizations are research outputs and do not establish clinical diagnostic accuracy.
