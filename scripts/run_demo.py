import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
from src.face_mesh.mediapipe_pipeline import FaceMeshDetector
from src.fatigue.ear import EARTracker
from src.fatigue.mar import MARTracker
from src.fatigue.perclos import PERCLOSTracker
from src.gaze.gaze_estimator import GazeEstimator

def load_configs():
    with open('configs/thresholds.yaml', 'r') as f:
        thresholds = yaml.safe_load(f)
    with open('configs/model_paths.yaml', 'r') as f:
        paths = yaml.safe_load(f)
    return thresholds, paths

def main():
    thresholds, paths = load_configs()

    print("Initializing components...")
    # Initialize Modules
    face_mesh = FaceMeshDetector(thresholds, paths)
    ear_tracker = EARTracker(thresholds)
    mar_tracker = MARTracker(thresholds)
    perclos_tracker = PERCLOSTracker(thresholds)
    gaze_estimator = GazeEstimator(thresholds)

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Module 1: Face Mesh & Pose
        result = face_mesh.process_frame(frame)

        # Modules 2 & 3: Fatigue geometries & Gaze
        result = ear_tracker.update(result)
        result = mar_tracker.update(result)
        result = perclos_tracker.update(result)
        result = gaze_estimator.update(result)

        # Lightweight Visualization Overlay
        if result['valid']:
            cv2.putText(frame, f"EAR: {result['ear']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {result['mar']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            perclos_val = result['perclos']
            p_text = f"{perclos_val:.2f}" if perclos_val is not None else "Buffering..."
            cv2.putText(frame, f"PERCLOS: {p_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            gaze = result.get('gaze_direction', 'N/A')
            cv2.putText(frame, f"Gaze: {gaze}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Head: P:{result['pitch']:.0f} Y:{result['yaw']:.0f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Overlay RED Alerts 
            y_pos = 190
            for alert in result['alerts']:
                cv2.putText(frame, f"ALERT: {alert}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_pos += 30
        else:
            cv2.putText(frame, "INVALID FRAME (No Face / Angle > Max)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Fatigue Detection Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == '__main__':
    main()
