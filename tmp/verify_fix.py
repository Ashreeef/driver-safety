import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.seatbelt.pipelines import Pipeline1, Pipeline3

def test_pipelines():
    yolo_v5_path = "weights/v1/best_github.pt"
    yolo_v8_path = "weights/v1/best.pt"
    cnn_path = "weights/v1/patch_cnn.pt"
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("--- Testing Pipeline 1 (YOLOv5 fallback) ---")
    try:
        p1 = Pipeline1(yolo_v5_path, cnn_path)
        print("Initialization successful (fallback triggered if needed)")
        ann, results = p1.process_frame(dummy_frame)
        print(f"Process frame successful. Results: {len(results)} detections")
    except Exception as e:
        print(f"Pipeline 1 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing Pipeline 3 (YOLOv8 modern) ---")
    try:
        p3 = Pipeline3(yolo_v8_path)
        print("Initialization successful")
        ann, results = p3.process_frame(dummy_frame)
        print("Process frame successful")
    except Exception as e:
        print(f"Pipeline 3 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipelines()
