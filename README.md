# Fatigue & Distraction Detection System
**ENSIA x Qareeb - Industrial Project**

Real-time driver monitoring using a single in-vehicle camera on Raspberry Pi 4.

## Supported Modules
- **Face Mesh & Head Pose**: Landmark tracking for orientation.
- **Geometric Fatigue**: EAR / MAR / PERCLOS computation.
- **Seatbelt Detection**: Core pipelines migrated and modularized.
- **Smoking Detection (New)**: Hybrid landmark/YOLO fusion engine.
- **Phone Detection (New)**: YOLO-based phone compliance detection with video event analysis.
- **Unified Safety Monitor (New)**: Integrated real-time monitoring of multiple hazards.

---

## Seatbelt Detection Pipeline


### Features
- **Pipeline 1**: YOLOv5 ROI Extraction + (CNN or YOLOv8n Classifier configurable via `seatbelt.yaml`).
- **Pipeline 2 (Recommended)**: MediaPipe Pose Landmarks for ROI + YOLO & CNN for Patch Classification + RANSAC Geometric Prior + EMA Temporal Smoothing.
- **Pipeline 3**: Direct YOLOv8n Frame Prediction + Configurable Smoother (EMA or Majority Vote).
- **Support for YOLOv5 & YOLOv8**: Robust model loading logic handles legacy and modern weights automatically.
- **Temporal Consistency**: Signal fusion modules (EMA, Majority Vote) to reduce flickering.

### Project Structure (Seatbelt)
- `src/compliance/seatbelt/`: Core modules (ROI extraction, classifier, geometric prior).
- `src/compliance/smoking/`: Modular smoking detection (Landmarks, Hybrid detector).
- `src/fusion/`: Signal smoothing (EMA, Majority Vote).
- `weights/`: Model weight storage (YOLO, MediaPipe).
- `scripts/unified_inference.py`: Integrated real-time hazard monitor.
- `scripts/smoking_inference.py`: Standalone smoking detection.
- `scripts/inference.py`: Individual seatbelt pipeline inference.
- `tests/test_seatbelt.py`: Automated unit tests for seatbelt utilities.

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

---

## 🚀 Running Inference

The project provides three primary inference scripts located in the `scripts/` directory. All scripts support real-time preview using the `--show` flag.

### 0. Phone Detection (Standalone)
Run phone-use detection on images or videos.

```bash
# Video inference with output
python scripts/phone_inference.py --video "data/test_video.mp4" \
    --weights "weights/v1/yolov8n_phone.pt" \
    --config "configs/phone.yaml" \
    --output "outputs/phone_output.mp4"

# Save per-frame event analysis JSON artifacts
python scripts/phone_inference.py --video "data/test_video.mp4" \
    --weights "weights/v1/yolov8n_phone.pt" \
    --save-analysis
```

Training utilities migrated from notebook flow:

```bash
# Prepare dataset (uses ROBOFLOW_API_KEY from .env)
python scripts/phone_train.py prepare --mode phone_only --dataset-version 1

# Train detector
python scripts/phone_train.py train --data dataset_phone/data.yaml

# Evaluate detector
python scripts/phone_train.py eval --weights runs/phone_detection/yolov8n_phone_v2/weights/best.pt --split test
```

### 1. Unified Safety Monitor (Production)
The primary entry point that runs **Smoking** and **Seatbelt** detection simultaneously.

```bash
# Run on video with live display and rate gate disabled
python scripts/unified_inference.py --video "data/test_video.mp4" --show --no-rate-gate

# Run and save to file with rate gate enabled (overriding YAML)
python scripts/unified_inference.py --video "data/test_video.mp4" --output "output.mp4" --rate-gate
```

**Custom Weights:**
```bash
python scripts/unified_inference.py --video 0 \
    --sb_yolo "weights/v1/best_github.pt" \
    --sb_clf  "weights/v1/best.pt" \
    --show
```

---

### 2. Smoking Detection (Standalone)
Dedicated script for smoking detection only.

```bash
python scripts/smoking_inference.py --video "data/test_video.mp4" --show
```

---

### 3. Seatbelt Detection (Individual Pipelines)
Runs specific seatbelt detection pipelines (1, 2, or 3).

```bash
# Pipeline 2 (Recommended)
python scripts/seatbelt_inference.py --type 2 \
    --video "data/test_video.mp4" \
    --yolo "weights/v1/best.pt" \
    --cnn "weights/v1/patch_cnn.pt" \
    --show --no-rate-gate
```

---

## ⚙️ Configuration
All systems are modular and controlled via YAML files in the `configs/` directory.

- **Smoking**: `configs/smoking.yaml` (Thresholds, Fusion Alpha, Temporal Window)
- **Seatbelt**: `configs/seatbelt.yaml` (ROI settings, Smoothers, Rate Gate)

---

## 🛠️ Automated Testing
To verify the integrity of seatbelt utilities:
```bash
pytest tests/test_seatbelt.py
```

## 📂 Project Structure
- `src/compliance/`: Core detector implementations.
  - `seatbelt/`: ROI extraction, Geometric prior, classifiers.
  - `smoking/`: MediaPipe landmarkers, Hybrid fusion detector.
- `src/fusion/`: Signal smoothing algorithms (EMA, Majority Vote).
- `scripts/`: CLI entry points for inference and evaluation.
- `configs/`: Centralized settings and thresholds.
- `weights/`: Pre-trained model weights.

---

## Overall Installation
```bash
pip install -r requirements.txt
```

## Docs
See `docs/reports/` for detailed technical specifications and meeting logs.
