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

        # Visualization Overlay
        y = 30
        def put(text, color=(0, 220, 0)):
            nonlocal y
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 28

        if result['valid']:
            cal = result.get('ear_calibrated', False)
            bl  = result.get('ear_baseline')
            cal_str = f"BL={bl:.3f}" if cal else f"CAL {calibrator.progress()*100:.0f}%"

            ear_val     = result['ear']
            mar_val     = result['mar']
            perclos_val = result['perclos']

            put(f"EAR: {ear_val:.3f} ({cal_str})")
            put(f"MAR: {mar_val:.3f}" if mar_val is not None else "MAR: --")
            put(f"PERCLOS: {perclos_val:.3f}" if perclos_val is not None else "PERCLOS: buffering")
            put(f"Gaze: {result.get('gaze_direction','--')}")
            put(f"Head P:{result['pitch']:.0f} Y:{result['yaw']:.0f} [{result['head_pose_method']}]")
            put(f"Yawns: {result.get('yawn_count', 0)}")

            for alert in result['alerts']:
                put(f"!! {alert}", color=(0, 0, 255))
        else:
            put("INVALID (no face / angle exceeded)", color=(0, 0, 255))
            put(f"P:{result['pitch']:.0f} Y:{result['yaw']:.0f}", color=(0, 140, 255))

        cv2.imshow('Fatigue Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == '__main__':
    main()
