# Fatigue & Distraction Detection System
**ENSIA x Qareeb - Industrial Project**

Real-time driver monitoring using a single in-vehicle camera on Raspberry Pi 4.

## Supported Modules
- **Face Mesh & Head Pose**: Landmark tracking for orientation.
- **Geometric Fatigue**: EAR / MAR / PERCLOS computation.
- **Seatbelt Detection (New)**: Core pipelines migrated and modularized.

---

## Seatbelt Detection Pipeline

I have successfully migrated the seatbelt detection logic into a production-ready structure.

### Features
- **Pipeline A**: YOLOv5/v8 Full-Frame Detection + MobileNetV3 Patch Classifier.
- **Pipeline B (Recommended)**: MediaPipe Pose Landmarks + YOLO High-Confidence Detection + RANSAC Geometric Prior + EMA Temporal Smoothing.
- **Support for YOLOv5 & YOLOv8**: Robust model loading logic handles legacy and modern weights automatically.
- **Temporal Consistency**: Signal fusion modules (EMA, Majority Vote) to reduce flickering.

### Project Structure (Seatbelt)
- `src/compliance/seatbelt/`: Core modules (ROI extraction, classifier, geometric prior).
- `src/fusion/`: Signal smoothing (EMA, Majority Vote).
- `weights/`: Model weight storage (Renamed to avoid namespace collisions with YOLO).
- `scripts/inference.py`: Main CLI for running detection.
- `tests/test_seatbelt.py`: Automated unit tests for core utilities.

### Setup
Ensure you have the required dependencies:
```bash
pip install ultralytics mediapipe torch torchvision opencv-python numpy
```

### Organize Weights
Due to namespace collisions with YOLO internal modules, the weights directory has been renamed from `models` to `weights`. Organize your files as follows:
```text
weights/
├── v1/
│   ├── best.pt              # YOLOv8 weights
│   ├── best_github.pt       # YOLOv5 weights
│   └── patch_cnn.pt         # MobileNetV3 patch classifier
└── mediapipe/
    └── pose_landmarker_lite.task
```

### Running Inference
The main entry point is `scripts/inference.py`.

**Pipeline A (YOLOv5/v8 Full-Frame)**:
```bash
python scripts/inference.py --type A --video your_video.mp4 --yolo weights/v1/best_github.pt --cnn weights/v1/patch_cnn.pt --show
```

**Pipeline B (Recommended: MediaPipe Pose + YOLO + RANSAC)**:
```bash
python scripts/inference.py --type B --video your_video.mp4 --yolo weights/v1/best.pt --cnn weights/v1/patch_cnn.pt --show
```

### Automated Testing
To run the automated tests for ROI extraction and signal smoothers:
```bash
pytest tests/test_seatbelt.py
```

---

## Overall Installation
```bash
pip install -r requirements.txt
```

## Docs
See docs/reports/ for meeting reports and planning documents.
