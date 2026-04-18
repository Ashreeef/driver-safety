from ultralytics import YOLO
import numpy as np
import cv2

def test_yolo():
    try:
        # Try to load the YOLOv8 model first as it's more likely to be compatible with YOLO class
        model_path = "weights/v1/best.pt"
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        
        # Test prediction
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Running prediction...")
        results = model(dummy_frame, verbose=False)
        print("Success for YOLOv8!")
        
    except Exception as e:
        print(f"Failed for YOLOv8: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Try YOLOv5 model with YOLO class
        model_path = "weights/v1/best_github.pt"
        print(f"\nLoading model from {model_path} with YOLO()...")
        model = YOLO(model_path)
        
        # Test prediction
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("Running prediction...")
        results = model(dummy_frame, verbose=False)
        print("Success for YOLOv5 with YOLO()!")
        
    except Exception as e:
        print(f"Failed for YOLOv5 with YOLO(): {e}")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    test_yolo()
