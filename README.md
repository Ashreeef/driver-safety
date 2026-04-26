# Driver Fatigue & Compliance Detection System

**ENSIA × Qareeb — Industrial Academic Project**

Real-time in-vehicle driver monitoring running entirely on a **Raspberry Pi 4 CPU** — no GPU, no cloud, fully offline. A single fixed RGB camera captures the driver's face. The system detects fatigue, distraction, and compliance violations simultaneously at a target of **≥ 15 FPS**.

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [System Architecture](#system-architecture)
- [Modules](#modules)
  - [Module 1 — Face Mesh & Head Pose](#module-1--face-mesh--head-pose)
  - [Module 2 — Fatigue Metrics (EAR / MAR / PERCLOS)](#module-2--fatigue-metrics-ear--mar--perclos)
  - [Module 3 — Gaze Estimation](#module-3--gaze-estimation)
  - [Module 5 — Compliance (Seatbelt / Smoking / Phone)](#module-5--compliance-seatbelt--smoking--phone)
  - [Module 6 — Alert Engine](#module-6--alert-engine)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Benchmarking](#benchmarking)
- [Configuration](#configuration)
- [Model Weights](#model-weights)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Branch Strategy](#branch-strategy)

---

## Overview

| Property | Value |
|---|---|
| Target hardware | Raspberry Pi 4 (4 GB RAM), CPU only |
| Camera | Single fixed RGB camera, driver-facing |
| Target FPS | ≥ 15 end-to-end |
| Network required | No — fully offline |
| Academic institution | ENSIA (École Nationale Supérieure d'Intelligence Artificielle, Algeria) |
| Industry partner | Qareeb |
| MVP deadline | Mid–end May 2026 |

---

## Team

| Member | Role |
|---|---|
| **Berbaoui Ashref** | Face mesh, EAR/MAR/PERCLOS, Gaze, Integration |
| **Hamza** | CNN-GRU deep fatigue model |
| **Imen** | Seatbelt detection, Smoking detection |
| **Yacine** | Phone detection |

---

## System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              Camera Frame (BGR)              │
                        └──────────────────────┬──────────────────────┘
                                               │
                              ┌────────────────▼────────────────┐
                              │   Module 1 — FaceMeshDetector   │
                              │   MediaPipe Tasks API (.task)    │
                              │   478 landmarks · head pose      │
                              │   EMA stabiliser · validity gate │
                              └──────┬───────────────────────────┘
                                     │  result_dict
               ┌─────────────────────┼──────────────────────┐
               │                     │                       │
    ┌──────────▼──────────┐  ┌───────▼──────────┐  ┌────────▼─────────┐
    │  Module 2 — Fatigue │  │ Module 3 — Gaze  │  │ Module 5 —       │
    │  EARTracker         │  │ GazeEstimator    │  │ ComplianceWorker  │
    │  MARTracker         │  │ Head-pose primary│  │ (daemon thread)   │
    │  PERCLOSTracker     │  │ Iris deviation   │  │ Seatbelt·Smoking  │
    │  EARCalibrator      │  │ Dir. stabiliser  │  │ Phone detector    │
    └──────────┬──────────┘  └───────┬──────────┘  └────────┬─────────┘
               │                     │                       │
               └─────────────────────▼───────────────────────┘
                                     │  result_dict (merged)
                          ┌──────────▼──────────┐
                          │  Module 6 —          │
                          │  AlertEngine         │
                          │  Escalation · Dedup  │
                          │  Priority sort       │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Overlay + Display   │
                          └─────────────────────┘
```

All modules share a single **`result_dict`** that is mutated in-place and passed downstream. The ComplianceWorker runs in a dedicated daemon thread so YOLO inference (~30–50 ms) does not block the main face-mesh loop.

---

## Modules

### Module 1 — Face Mesh & Head Pose

**File:** `src/face_mesh/mediapipe_pipeline.py`

Uses the **MediaPipe Tasks API** (not the legacy `mp.solutions.face_mesh`) with a local `.task` file for offline deployment. Outputs 478 3D landmarks normalized to `[0, 1]`.

- **EMA landmark stabilizer** (`alpha = 0.4`) applied every frame to reduce jitter. Resets after 1 second of no face.
- **Head pose** computed via `solvePnP` (6-point model). Falls back to `facial_transformation_matrix` if `|yaw| > 90°` or solvePnP fails.
- **Validity gate:** `valid = False` when `|yaw| > 60°` or `|pitch| > 40°`. Downstream modules skip computation when `valid` is False.
- Camera mount offsets (`camera_yaw_offset`, `camera_pitch_offset`) subtracted before the validity check.

### Module 2 — Fatigue Metrics (EAR / MAR / PERCLOS)

**Files:** `src/fatigue/ear.py`, `src/fatigue/mar.py`, `src/fatigue/perclos.py`

#### Eye Aspect Ratio (EAR)

```
EAR = ( ||P2−P6|| + ||P3−P5|| ) / ( 2 × ||P1−P4|| )
```

Computed for both eyes and averaged. Scale-invariant ratio.

**EARCalibrator** collects open-eye EAR samples for the first 10 seconds:
- Rejects samples outside `[0.10, 0.65]`
- Derives per-driver thresholds: `alert = baseline × 0.75`, `perclos = baseline × 0.27`
- Falls back to YAML defaults before calibration completes

**Alerts generated:**
| Alert | Condition |
|---|---|
| `Drowsiness (EAR)` | Smoothed EAR below threshold for 4 consecutive frames |
| `Fatigue (EAR Trend)` | First-half mean of 30-frame buffer drops > 0.06 vs second half |

#### Mouth Aspect Ratio (MAR)

```
MAR = ||M_top − M_bottom|| / ||M_left − M_right||
```

Uses landmarks 13/14 (extreme lip centers) for maximum vertical excursion (~2× signal vs corner points).

**Alerts generated:**
| Alert | Condition |
|---|---|
| `Yawning (MAR)` | MAR > 0.50 sustained for ≥ 2.0 seconds |
| `Fatigue (Yawn Frequency)` | 3+ confirmed yawns in a rolling 5-minute window |

#### PERCLOS

```
PERCLOS = closed_frames_in_window / window_size
```

- 60-second rolling window at 15 FPS (900 frames)
- Returns `None` for the first 60 seconds (buffering — by design)
- Threshold is calibrated per-driver, not hardcoded

**Alert:** `Drowsiness (PERCLOS)` when PERCLOS > 0.15

### Module 3 — Gaze Estimation

**File:** `src/gaze/gaze_estimator.py`

Two-signal fused approach with per-driver calibration:

**1. Head-pose primary** (always active, no calibration needed):
- `|yaw| > 20°` → left / right
- `pitch > 15°` → down
- `pitch < −10°` → up
- Head near-forward → fall through to iris

**2. Iris deviation from calibrated neutral** (head near-forward only):
- Calibration collects samples where `|yaw| < 10°` and `|pitch| < 8°`
- Neutral computed as median of calibration window
- `dh = h_ratio − neutral_h`, `dv = v_ratio − neutral_v`
- Direction set by deviation crossing configurable thresholds

**Direction stabilizer:** new direction must persist for 5 consecutive frames (~333 ms at 15 FPS) before `_stable_direction` changes.

**Alert:** `Distraction (Gaze)` when stable direction ≠ "forward" for ≥ 2.0 seconds.

Overlay indicator: `[H]` = head-driven, `[I]` = iris-driven, `[?]` = pre-calibration.

### Module 5 — Compliance (Seatbelt / Smoking / Phone)

All compliance modules run inside a **ComplianceWorker daemon thread** to keep the main loop unblocked.

#### Seatbelt Detection

**Files:** `src/compliance/seatbelt/`

Three selectable pipelines (default: Pipeline 2):

| Pipeline | Description |
|---|---|
| **Pipeline 1** | YOLOv5s ROI extractor → MobileNetV3 CNN classifier. Has loader issues, not recommended. |
| **Pipeline 2** *(default)* | MediaPipe Pose ROI → YOLOv8n → MobileNetV3 CNN → RANSAC geometric prior → EMA temporal smoothing |
| **Pipeline 3** | YOLOv8n direct frame inference. Simpler, less robust. |

A trained **BiLSTM temporal smoother** (`seatbelt_bilstm.pt`) is available but not yet wired into any pipeline (post-MVP item).

#### Smoking Detection

**Files:** `src/compliance/smoking/`

Hybrid detector fusing two independent branches:

| Branch | Method | Weight |
|---|---|---|
| A — Landmark | Hand proximity score + elbow angle + velocity penalty | 10% |
| B — Detection | YOLOv8 ONNX (`smoking_yolov8.onnx`, class `Smooking`) | 90% |

`s_F = 0.10 × s_L + 0.90 × s_D`

Temporal confirmation: sliding 8-frame window, confirm at 5+, hysteresis clears below 0.30.

The LandmarkExtractor runs in **dual-mode**: when integrated into the main pipeline it reuses face landmarks from Module 1 (no extra inference). Standalone mode runs all three MediaPipe models.

#### Phone Detection

**File:** `src/compliance/phone.py`

YOLOv8n fine-tuned on phone dataset. Class names matched against `{"phone", "cell phone", "cellphone", "mobile"}` (case-insensitive) — robust to varied label conventions. Steering wheel detections suppressed via `HIDDEN_CLASSES`.

- Confidence threshold: 0.25 (lower than seatbelt/smoking — phones can be partially occluded)
- Output: `PhoneFrameResult` dataclass with `alert_triggered`, `n_phones`, `detections`

### Module 6 — Alert Engine

**File:** `src/fusion/alert_engine.py`

Called every frame after all modules have run. Responsibilities:

1. **Compliance bridge** — converts booleans to alert strings:
   - `seatbelt_detected = False` → `'Seatbelt OFF'` (suppressed for first 5 s warmup)
   - `smoking_detected = True` → `'Smoking detected'`
   - `phone_detected = True` → `'Phone detected'`

2. **Cross-module escalation** — if 2 or more of the following fire simultaneously, they are replaced with a single `'CRITICAL FATIGUE'` alert:
   - `Drowsiness (EAR)`, `Drowsiness (PERCLOS)`, `Fatigue (EAR Trend)`, `Fatigue (Yawn Frequency)`

3. **Priority sort** (highest first):

| Alert | Priority |
|---|---|
| CRITICAL FATIGUE | 10 |
| Drowsiness (EAR) | 5 |
| Drowsiness (PERCLOS) | 5 |
| Fatigue (EAR Trend) | 4 |
| Fatigue (Yawn Frequency) | 3 |
| Yawning (MAR) | 2 |
| Distraction (Gaze) | 2 |
| Seatbelt OFF | 2 |
| Smoking detected | 2 |
| Phone detected | 2 |

---

## Installation

### Runtime (Raspberry Pi + laptop)

```bash
git clone https://github.com/Ashreeef/driver-safety.git
cd driver-safety
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Development tools (laptop / cloud only — do NOT install on RPi)

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` contains training utilities (`roboflow`, `python-dotenv`, `memory_profiler`, `psutil`) that are not needed at runtime.

### Verify model weights

```bash
python -c "
import yaml, os
paths = yaml.safe_load(open('configs/model_paths.yaml'))
checks = [
    paths['mediapipe']['face_landmarker'],
    paths['mediapipe']['hand_landmarker'],
    paths['mediapipe']['pose_landmarker'],
    paths['seatbelt']['yolo_p2'],
    paths['seatbelt']['classifier'],
    paths['smoking']['yolo'],
    paths['phone']['yolo'],
]
for p in checks:
    print('OK     ' if os.path.exists(p) else 'MISSING', p)
"
```

---

## How to Run

```bash
# Webcam — all modules (default)
python scripts/run_demo.py

# Video file as source
python scripts/run_demo.py --source path/to/video.mp4

# Save annotated output video
python scripts/run_demo.py --source video.mp4 --output out.mp4

# Specific camera index
python scripts/run_demo.py --source 1

# Disable all compliance modules (seatbelt + smoking + phone)
python scripts/run_demo.py --no-compliance

# Disable individual compliance modules
python scripts/run_demo.py --no-smoking
python scripts/run_demo.py --no-phone

# Choose seatbelt pipeline (default: 2)
python scripts/run_demo.py --seatbelt-pipeline 1   # YOLOv5s ROI + CNN
python scripts/run_demo.py --seatbelt-pipeline 2   # Pose + YOLOv8n + CNN (recommended)
python scripts/run_demo.py --seatbelt-pipeline 3   # Direct YOLOv8n
```

**Keyboard shortcuts while running:**
- `q` — quit cleanly
- `r` — reset gaze calibration

### Standalone inference scripts

```bash
# Phone detection on a video file
python scripts/phone_inference.py --video video.mp4 \
    --weights weights/v1/yolov8n_phone.pt \
    --config configs/phone.yaml

# Smoking detection standalone
python scripts/smoking_inference.py --video video.mp4

# Seatbelt detection standalone
python scripts/seatbelt_inference.py --type 2 --video video.mp4
```

---

## Benchmarking

`scripts/evaluate.py` measures per-component latency and reports FPS. Designed for RPi benchmarking.

```bash
# Full pipeline — 300 frames after warmup, display window, timing report
python scripts/evaluate.py --frames 300

# Video file input with annotated output
python scripts/evaluate.py --source video.mp4 --save-video out.mp4

# Core modules only (no YOLO), headless
python scripts/evaluate.py --frames 300 --no-compliance --no-display

# Save timing report to file
python scripts/evaluate.py --frames 300 --output report.txt
```

**Sample report output:**

```
================================================================================
  FATIGUE DETECTION — LATENCY BENCHMARK REPORT
================================================================================
  Frames processed : 300
  Wall time        : 20.1s
  End-to-end FPS   : 14.9  (target ≥15)

  ── Main-thread modules (synchronous) ───────────────────────────────────────
  Face Mesh (MediaPipe)           mean=  48.3ms  p50=  46.1ms  p95=  62.4ms
  EAR + EARTracker                mean=   0.3ms  ...
  MAR + MARTracker                mean=   0.2ms  ...
  PERCLOS                         mean=   0.1ms  ...
  Gaze Estimator                  mean=   1.1ms  ...
  Alert Engine                    mean=   0.0ms  ...
  ──────────────────────────────────────────────────────────────────────────
  Main-thread subtotal            mean=  50.0ms  →  20.0 FPS budget

  ── Compliance worker thread (asynchronous) ──────────────────────────────
  Seatbelt Pipeline      [async]  mean= 118.4ms  ...
  Smoking Detector       [async]  mean=  92.1ms  ...
  Phone Detector         [async]  mean=  38.7ms  ...
================================================================================
```

Compliance modules are `[async]` — they run in a daemon thread and do **not** subtract from the main-loop FPS.

---

## Configuration

All tunable parameters live in `configs/`. **No magic numbers in source code.**

### `configs/thresholds.yaml` — global parameters

```yaml
# Head pose validity gate
head_yaw_max: 60
head_pitch_max: 40
camera_yaw_offset: 0
camera_pitch_offset: 0
landmark_ema_alpha: 0.4

# EAR
ear_threshold: 0.20
ear_consec_frames: 4
ear_calibration_seconds: 10
ear_closure_ratio: 0.75          # alert_threshold = baseline × this
ear_perclos_ratio: 0.27          # perclos_threshold = baseline × this
ear_trend_drop_threshold: 0.06
ear_smooth_frames: 10

# MAR / yawn
mar_threshold: 0.50
yawn_min_seconds: 2.0
yawn_window_seconds: 300
yawn_count_alert: 3

# PERCLOS
perclos_window_seconds: 60
perclos_alert_level: 0.15

# Gaze
gaze_alert_seconds: 2.0
gaze_confirm_frames: 5
gaze_h_threshold: 0.12
gaze_v_up_threshold: 0.10
gaze_v_down_threshold: 0.13

# Camera
camera_source: 0
camera_width: 640
camera_height: 480
camera_fps: 15
```

### `configs/model_paths.yaml` — weight file locations

```yaml
mediapipe:
  face_landmarker: "models/mediapipe/face_landmarker.task"
  hand_landmarker: "models/mediapipe/hand_landmarker.task"
  pose_landmarker: "models/mediapipe/pose_landmarker_lite.task"

seatbelt:
  yolo_p1:    "weights/v1/seatbelt_yolov5s_roi.pt"   # Pipeline 1
  yolo_p2:    "weights/v1/seatbelt_yolov8n.pt"        # Pipeline 2/3 (default)
  classifier: "weights/v1/seatbelt_mobilenetv3.pt"    # Shared CNN patch classifier
  bilstm:     "weights/v1/seatbelt_bilstm.pt"         # BiLSTM smoother (not wired yet)

smoking:
  yolo: "weights/v1/smoking_yolov8.onnx"

phone:
  yolo: "weights/v1/yolov8n_phone.pt"
```

---

## Model Weights

| File | Architecture | Used by |
|---|---|---|
| `seatbelt_yolov8n.pt` | YOLOv8n, 2-class (`Seat_Belt` / `Without_Seat_Belt`) | Seatbelt Pipeline 2 & 3 |
| `seatbelt_yolov5s_roi.pt` | YOLOv5s, 2-class | Seatbelt Pipeline 1 ROI extractor |
| `seatbelt_mobilenetv3.pt` | MobileNetV3-Small, 2-class | Seatbelt patch classifier (P1 + P2) |
| `seatbelt_bilstm.pt` | BiLSTM, 2-class | Trained — not yet wired (post-MVP) |
| `seatbelt_yolov10n_unused.pt` | YOLOv10n | Evaluated, not good — unused |
| `smoking_yolov8.onnx` | YOLOv8 ONNX, class `Smooking` | Smoking detector YOLO branch |
| `yolov8n_phone.pt` | YOLOv8n, fine-tuned | Phone detector (active) |
| `yolo26n_phone.pt` | Experimental | Not referenced — kept for reference |

> All weights use **Apache 2.0** compatible training pipelines. AGPL-3.0 (YOLOv5 training code) is commercially blocked for Qareeb — only inference use of YOLOv5-format weights is kept, loaded via `torch.hub`.

---

## Project Structure

```
fatigue-detection/
├── configs/
│   ├── thresholds.yaml          # ALL tunable parameters
│   ├── model_paths.yaml         # paths to all model weight files
│   ├── seatbelt.yaml            # seatbelt pipeline config
│   ├── smoking.yaml             # smoking detector config
│   └── phone.yaml               # phone detector config
├── weights/
│   └── v1/
│       ├── seatbelt_yolov8n.pt
│       ├── seatbelt_yolov5s_roi.pt
│       ├── seatbelt_mobilenetv3.pt
│       ├── seatbelt_bilstm.pt
│       ├── seatbelt_yolov10n_unused.pt
│       ├── smoking_yolov8.onnx
│       ├── yolov8n_phone.pt
│       └── yolo26n_phone.pt
├── models/
│   └── mediapipe/
│       ├── face_landmarker.task
│       ├── hand_landmarker.task
│       └── pose_landmarker_lite.task
├── src/
│   ├── face_mesh/
│   │   ├── mediapipe_pipeline.py   # Module 1 — FaceMeshDetector
│   │   ├── head_pose.py            # solvePnP + transform_matrix fallback
│   │   └── landmark_utils.py       # ALL landmark index constants
│   ├── fatigue/
│   │   ├── ear.py                  # EARCalibrator + EARTracker
│   │   ├── mar.py                  # MARTracker
│   │   ├── perclos.py              # PERCLOSTracker
│   │   └── dl_model.py             # Module 4 stub (Hamza — deferred)
│   ├── gaze/
│   │   └── gaze_estimator.py       # GazeEstimator
│   ├── compliance/
│   │   ├── adapters.py             # seatbelt_to_result / smoking_to_result / phone_to_result
│   │   ├── phone.py                # PhoneDetector + VideoAnalyzer
│   │   ├── smoking/
│   │   │   ├── detector.py         # SmokingDetector (hybrid YOLO + landmark)
│   │   │   └── landmarks.py        # LandmarkExtractor (dual-mode)
│   │   └── seatbelt/
│   │       ├── pipelines.py        # Pipeline1 / Pipeline2 / Pipeline3
│   │       ├── classifier.py       # MobileNetV3 patch classifier
│   │       ├── roi.py              # ROIExtractor (MediaPipe Pose)
│   │       └── geometric.py        # RANSAC seatbelt line prior
│   └── fusion/
│       └── alert_engine.py         # Module 6 — AlertEngine
├── scripts/
│   ├── run_demo.py                 # Main entry point
│   ├── evaluate.py                 # Per-component latency benchmarking
│   ├── phone_inference.py          # Standalone phone detection
│   ├── phone_train.py              # Phone dataset prep + YOLO training
│   ├── smoking_inference.py        # Standalone smoking detection
│   └── seatbelt_inference.py       # Standalone seatbelt detection
├── tests/
│   ├── test_ear.py
│   ├── test_perclos.py
│   ├── test_mediapipe.py
│   ├── test_gaze.py
│   └── test_seatbelt.py
├── docs/
│   └── progress_reports/
├── requirements.txt                # Runtime deps — install on RPi
└── requirements-dev.txt            # Training/dev tools — laptop only
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Individual test suites
pytest tests/test_ear.py
pytest tests/test_perclos.py
pytest tests/test_gaze.py
pytest tests/test_seatbelt.py
pytest tests/test_mediapipe.py

# With coverage report
pytest --cov=src tests/
```

---

## Branch Strategy

```
main   ← stable releases only, protected, no direct pushes
dev    ← integration branch, all features merge here first
feat/* ← feature branches, PR → dev
```

Commit format: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Scopes: `face-mesh`, `fatigue`, `gaze`, `compliance`, `fusion`, `utils`, `config`

---

## Open Items (MVP priority order)

1. **Seatbelt model accuracy** — Pipeline 2 integrated; detection quality needs evaluation on labelled video
2. **Smoking model accuracy** — ONNX model needs quantitative benchmarking
3. **Phone model accuracy** — `yolov8n_phone.pt` needs evaluation on driver-angle footage
4. **RPi FPS benchmark** — run `scripts/evaluate.py` on Pi, measure per-component latency
5. **Roll validity gate** — add soft `|roll| > 25°` limit in `mediapipe_pipeline.py`
6. **Gaze threshold tuning** — record 50-frame labelled test, tune `gaze_h_threshold` / `gaze_v_*_threshold`
7. **Camera offset calibration script** — 30s forward-gaze recording → compute mean pitch/yaw offsets
8. **BiLSTM integration** *(optional)* — wire `seatbelt_bilstm.pt` into Pipeline 2; key already in `model_paths.yaml`
