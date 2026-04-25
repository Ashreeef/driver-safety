import cv2
import argparse
import time
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.smoking.landmarks import LandmarkExtractor
from src.compliance.smoking.detector import SmokingDetector

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def draw_hud(frame, res, frame_idx, fusion_score, temporal_conf):
    H, W = frame.shape[:2]
    out = frame.copy()
    
    # Top Banner
    banner_h = 70
    overlay = out.copy()
    color = (0, 0, 180) if res.alert_active else (20, 150, 20)
    cv2.rectangle(overlay, (0, 0), (W, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)
    
    # Alert Text
    label = "⚠ SMOKING DETECTED" if res.alert_active else "✓ SMOKING: CLEAR"
    cv2.putText(out, label, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Bottom HUD
    cv2.rectangle(out, (0, H - 35), (W, H), (10, 10, 10), -1)
    stats = f"Fusion: {fusion_score:.2f} | TempConf: {temporal_conf:.2f} | Frame: {frame_idx}"
    cv2.putText(out, stats, (15, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return out

def run_smoking_inference(video_path, sm_weights, config_path, mp_paths_path, output_path=None, show=False):
    config = load_config(config_path)
    mp_paths = load_config(mp_paths_path).get('mediapipe', {})

    print("  Loading Smoking Subsystem (standalone mode — internal face detector)...")
    extractor = LandmarkExtractor(
        config.get('landmarks', {}),
        model_paths=mp_paths,
        use_external_face_landmarks=False,
    )
    detector = SmokingDetector(config, extractor, sm_weights)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    W, H = int(cap.get(3)), int(cap.get(4))
    FPS = cap.get(5) or 30
    
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
        print(f"💾 Recording to {output_path}")

    frame_idx = 0
    t_start = time.time()

    # Window resizing logic
    win_name = "Smoking Detection"
    display_w = 1024  # Standard display width for "smaller" window
    display_h = int(display_w * (H / W))

    try:
        if show:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, display_w, display_h)

        while True:
            ret, frame = cap.read()
            if not ret: break

            res = detector.process(frame)
            
            # Annotated frame
            annotated = draw_hud(frame, res, frame_idx, res.fusion_score, res.temporal_conf)
            
            # Landmarks
            lm = res.landmarks or {}
            if "right_wrist" in lm:
                wx, wy = int(lm["right_wrist"][0] * W), int(lm["right_wrist"][1] * H)
                cv2.circle(annotated, (wx, wy), 8, (0, 255, 130), -1)
            if "mouth_centre" in lm:
                mx, my = int(lm["mouth_centre"][0] * W), int(lm["mouth_centre"][1] * H)
                cv2.circle(annotated, (mx, my), 5, (0, 200, 255), -1)

            # Detections
            for name, conf, bbox in res.detections:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(annotated, f"{name} {conf:.0%}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            if writer: writer.write(annotated)
            
            if show:
                cv2.imshow(win_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                fps = frame_idx / (time.time() - t_start)
                print(f"  Frame {frame_idx} | FPS: {fps:.1f}")

    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"✅ Finished. Total frames: {frame_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoking Detection Standalone Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--weights", type=str, default="weights/smoking/best.onnx", help="Path to smoking YOLO weights")
    parser.add_argument("--config", type=str, default="configs/smoking.yaml", help="Path to smoking YAML config")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    
    parser.add_argument("--model-paths", type=str, default="configs/model_paths.yaml",
                        help="Path to model_paths.yaml")

    args = parser.parse_args()

    run_smoking_inference(args.video, args.weights, args.config, args.model_paths,
                          args.output, args.show)
