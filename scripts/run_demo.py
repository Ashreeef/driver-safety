import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
from src.face_mesh.mediapipe_pipeline import FaceMeshDetector
from src.fatigue.ear import EARCalibrator, EARTracker
from src.fatigue.mar import MARTracker
from src.fatigue.perclos import PERCLOSTracker
from src.gaze.gaze_estimator import GazeEstimator

def load_configs():
    with open('configs/thresholds.yaml') as f:
        thresholds = yaml.safe_load(f)
    with open('configs/model_paths.yaml') as f:
        paths = yaml.safe_load(f)
    return thresholds, paths

def main():
    thresholds, paths = load_configs()
    fps = thresholds.get('camera_fps', 15)

    print("Initializing components...")
    # Initialize Modules
    face_mesh    = FaceMeshDetector(thresholds, paths)
    calibrator   = EARCalibrator(thresholds, fps=fps)
    ear_tracker  = EARTracker(thresholds, calibrator)
    mar_tracker  = MARTracker(thresholds)
    perclos      = PERCLOSTracker(thresholds, calibrator, fps=fps)
    gaze         = GazeEstimator(thresholds)

    source = thresholds.get('camera_source', 0)
    cap    = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  thresholds.get('camera_width',  640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, thresholds.get('camera_height', 480))
    cap.set(cv2.CAP_PROP_FPS,          fps)

    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Module 1: Face Mesh & Pose
        result = face_mesh.process_frame(frame)

        # Modules 2 & 3: Fatigue geometries & Gaze
        result = ear_tracker.update(result)
        result = mar_tracker.update(result)
        result = perclos.update(result)
        result = gaze.update(result)

        # Visualization Overlay (compact, grouped, and easy to scan)
        h, w = frame.shape[:2]
        panel_w = 245
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y = 28
        step = 24

        def put(text, color=(190, 210, 190), bold=False):
            nonlocal y
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2 if bold else 1,
                cv2.LINE_AA,
            )
            y += step

        put("DRIVER MONITOR", color=(100, 200, 255), bold=True)
        put("-" * 16, color=(70, 70, 70))

        if result['valid']:
            cal = result.get('ear_calibrated', False)
            bl = result.get('ear_baseline')
            ear_val = result.get('ear')
            mar_val = result.get('mar')
            perclos_val = result.get('perclos')

            if cal and bl is not None:
                cal_str = f"BL={bl:.3f}"
            else:
                cal_str = f"CAL {calibrator.progress() * 100:.0f}%"

            if ear_val is None:
                ear_color = (170, 170, 170)
                put("EAR  --", color=ear_color)
            else:
                ear_color = (0, 80, 255) if ear_val < calibrator.alert_threshold else (100, 220, 100)
                put(f"EAR  {ear_val:.3f}  {cal_str}", color=ear_color)

            mar_alert_thr = thresholds.get('mar_threshold', 0.5)
            mar_color = (0, 80, 255) if (mar_val is not None and mar_val > mar_alert_thr) else (100, 220, 100)
            put(f"MAR  {mar_val:.3f}" if mar_val is not None else "MAR  --", color=mar_color)

            if perclos_val is None:
                put("PERCLOS  buffering...", color=(180, 180, 60))
            else:
                pct = perclos_val * 100
                if pct < 8:
                    pc_color, label = (100, 220, 100), "OK"
                elif pct < 15:
                    pc_color, label = (0, 165, 255), "MILD"
                elif pct < 25:
                    pc_color, label = (0, 80, 255), "MOD"
                else:
                    pc_color, label = (0, 0, 255), "HIGH"
                put(f"PERCLOS  {pct:.1f}%  {label}", color=pc_color)

            put(f"Yawns  {result.get('yawn_count', 0)}", color=(180, 180, 180))
            put("-" * 16, color=(70, 70, 70))

            gaze_dir = result.get('gaze_direction', '--')
            gaze_cal = result.get('gaze_calibrated', False)
            gaze_prog = result.get('gaze_calib_progress', 0.0)
            gaze_h = result.get('gaze_h_ratio')
            gaze_v = result.get('gaze_v_ratio')

            if not gaze_cal:
                put(f"GAZE CAL  {gaze_prog * 100:.0f}%", color=(180, 180, 60))
                put("Look straight ahead", color=(150, 150, 150))
            else:
                gaze_color = (100, 220, 100) if gaze_dir == 'forward' else (0, 80, 255)
                put(f"Gaze  {gaze_dir}", color=gaze_color)
                if gaze_h is not None and gaze_v is not None:
                    put(f"h={gaze_h:.3f}  v={gaze_v:.3f}", color=(150, 150, 150))

            put(
                f"Head  P:{result['pitch']:.0f} Y:{result['yaw']:.0f} "
                f"[{result['head_pose_method'][:3]}]",
                color=(160, 160, 160),
            )

            alerts = result.get('alerts', [])
            if alerts:
                put("-" * 16, color=(70, 70, 70))
                put("ALERTS", color=(0, 120, 255), bold=True)
                for alert in alerts[:3]:
                    put(f"! {alert}", color=(0, 0, 255))
        else:
            put("NO FACE / ANGLE EXCEEDED", color=(0, 0, 255), bold=True)
            put(f"Pitch:{result['pitch']:.0f}  Yaw:{result['yaw']:.0f}", color=(0, 140, 255))

        cv2.imshow('Fatigue Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == '__main__':
    main()
