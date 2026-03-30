# Meeting: March 26, 2025

## Confirmed Decisions

| Topic | Decision |
| :--- | :--- |
| **Edge hardware** | Raspberry Pi 4. All models must run at ≥15 FPS on this device. |
| **Face mesh** | MediaPipe confirmed. Supervisor uses it and downloaded the model file locally (likely `face_landmarker.task` - the Tasks API bundle for offline use). We do not need to replicate this exactly but should be aware of offline model loading. |
| **Camera / IR** | No IR for now. IR camera ordered, pending delivery. Treat as a known limitation. Ignore edge cases caused by lighting at this stage. |
| **YOLO licensing** | Use freely in development. Supervisor's exact framing: fix the data first, find best results - then changing models is easy, one variable, all others fixed. |
| **RTMDet** | Worth exploring. Supervisor mentioned it positively. Apache 2.0 licensed, competitive with YOLO at edge sizes. Needs a dedicated research note from the compliance team before Thursday. |
| **Task structure** | Two parallel tracks: (1) Distraction (gaze vector estimation) and (2) Drowsiness (EAR/PERCLOS/MAR). Compliance runs as a third parallel workstream. |
| **Head pose** | Explicitly flagged as a challenge. Not a side detail - a core technical problem that must be solved as a dependency for both gaze and for making EAR reliable under head rotation. |
| **Data** | Supervisor will send. Not yet received as of March 27. Follow up by April 1 if not received. |
| **Timeline** | MVP by mid-to-end May 2025 for academic validation. Project then continues as Qareeb internship. |
