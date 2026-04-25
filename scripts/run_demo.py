import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import threading
import cv2
import yaml
from src.face_mesh.mediapipe_pipeline import FaceMeshDetector
from src.fatigue.ear import EARCalibrator, EARTracker
from src.fatigue.mar import MARTracker
from src.fatigue.perclos import PERCLOSTracker
from src.gaze.gaze_estimator import GazeEstimator


def _play_alert_sound():
    """Fire a short beep in a daemon thread — non-blocking."""
    if sys.platform == 'win32':
        import winsound
        winsound.Beep(1000, 350)
    else:
        os.system('aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || beep 2>/dev/null || true')


def trigger_audio(alerts, prev_alerts, audio_enabled):
    """Beep once whenever the active alert set gains a new entry."""
    if not audio_enabled:
        return
    if set(alerts) - set(prev_alerts):
        t = threading.Thread(target=_play_alert_sound, daemon=True)
        t.start()

def load_configs():
    with open('configs/thresholds.yaml') as f:
        thresholds = yaml.safe_load(f)
    with open('configs/model_paths.yaml') as f:
        paths = yaml.safe_load(f)
    return thresholds, paths

def main():
    parser = argparse.ArgumentParser(description='Driver fatigue detection demo')
    parser.add_argument('--source', default=None,
                        help='Video source: camera index (0, 1, …) or path to video file')
    parser.add_argument('--output', default=None,
                        help='Path for annotated output video (e.g. out.mp4). Omit to disable.')
    args = parser.parse_args()

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

    # CLI --source overrides thresholds.yaml; fall back to yaml value then 0
    if args.source is not None:
        raw = args.source
        source = int(raw) if raw.isdigit() else raw
    else:
        source = thresholds.get('camera_source', 0)
    cap    = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  thresholds.get('camera_width',  640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, thresholds.get('camera_height', 480))
    cap.set(cv2.CAP_PROP_FPS,          fps)

    # Output video writer (None when --output not specified)
    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, real_fps, (w, h))
        print(f"Recording annotated output → {args.output}")

    audio_enabled = thresholds.get('alerts_audio', True)
    prev_alerts: list = []

    print("Press 'q' to quit  |  'r' to reset gaze calibration.")
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

            gaze_dir  = result.get('gaze_direction', '--')
            gaze_cal  = result.get('gaze_calibrated', False)
            gaze_prog = result.get('gaze_calib_progress', 0.0)
            gaze_h    = result.get('gaze_h_ratio')
            gaze_v    = result.get('gaze_v_ratio')

            # Determine which signal is driving gaze (head vs iris)
            head_yaw_thresh   = thresholds.get('gaze_head_yaw_threshold', 20.0)
            head_pd_thresh    = thresholds.get('gaze_head_pitch_down_threshold', 15.0)
            head_pu_thresh    = thresholds.get('gaze_head_pitch_up_threshold', 10.0)
            yaw_abs  = abs(result.get('yaw', 0.0))
            pitch_v  = result.get('pitch', 0.0)
            head_driven = (yaw_abs > head_yaw_thresh or
                           pitch_v > head_pd_thresh or
                           pitch_v < -head_pu_thresh)
            gaze_signal = "H" if head_driven else ("I" if gaze_cal else "?")

            if not gaze_cal and not head_driven:
                put(f"GAZE CAL  {gaze_prog * 100:.0f}%", color=(180, 180, 60))
                put("Look straight ahead", color=(150, 150, 150))
            else:
                gaze_color = (100, 220, 100) if gaze_dir == 'forward' else (0, 80, 255)
                put(f"Gaze  {gaze_dir}  [{gaze_signal}]", color=gaze_color)
                if gaze_h is not None and gaze_v is not None and not head_driven:
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

        alerts = result.get('alerts', [])
        trigger_audio(alerts, prev_alerts, audio_enabled)
        prev_alerts = list(alerts)

        if writer is not None:
            writer.write(frame)

        cv2.imshow('Fatigue Detection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            gaze.reset_calibration()
            print("[run_demo] Gaze calibration reset.")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video → {args.output}")
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == '__main__':
    main()
