
# DFUVision Web Application

DFUVision is a Flask-based web application for diabetic
foot ulcer localization using the final trained YOLO26n model.

## Features

- Upload JPG, JPEG, and PNG foot images.
- Detect suspected diabetic foot ulcer regions.
- Display bounding boxes and confidence scores.
- Display the total number of detected regions.
- Download images with predicted bounding boxes.
- Optionally inspect a user-selected image region.

## Model Configuration

- Model: YOLO26n
- Input resolution: 512 x 512
- Standard confidence threshold: 0.20
- Model checkpoint: ../models/best.pt

## Technology

- Python
- Flask
- Ultralytics YOLO
- PyTorch
- HTML, CSS, and JavaScript

This application is intended for research demonstration
and has not been clinically validated.
