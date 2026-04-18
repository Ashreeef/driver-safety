import cv2
import numpy as np
import argparse
import time
import os
import sys
import yaml
import copy
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.seatbelt.pipelines import Pipeline1, Pipeline2, Pipeline3

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_grid(frames, titles, grid_size=(2, 3), cell_size=(640, 360)):
    rows, cols = grid_size
    canvas_w = cols * cell_size[0]
    canvas_h = rows * cell_size[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    for i, frame in enumerate(frames):
        if i >= rows * cols:
            break
        
        r = i // cols
        c = i % cols
        
        # Resize frame to cell size
        resized = cv2.resize(frame, cell_size)
        
        # Add title overlay
        cv2.rectangle(resized, (0, 0), (cell_size[0], 30), (0, 0, 0), -1)
        cv2.putText(resized, titles[i], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = r * cell_size[1]
        x_offset = c * cell_size[0]
        canvas[y_offset:y_offset+cell_size[1], x_offset:x_offset+cell_size[0]] = resized
        
    return canvas

def run_comparison(video_path, yolo_v5, yolo_v8, cnn_weights, config_path, output_path=None, show=False):
    base_config = load_config(config_path)
    device = 'cpu'
    
    # Define 5 pipelines
    print("Initializing 5 Pipeline Variations...")
    
    # 1. Pipeline 1 (CNN)
    cfg1_cnn = copy.deepcopy(base_config)
    cfg1_cnn.setdefault('pipelines', {}).setdefault('pipeline1', {})['classifier_type'] = 'cnn'
    p1_cnn = Pipeline1(yolo_v5, cnn_weights, device=device, config=cfg1_cnn)
    
    # 2. Pipeline 1 (YOLO)
    cfg1_yolo = copy.deepcopy(base_config)
    cfg1_yolo.setdefault('pipelines', {}).setdefault('pipeline1', {})['classifier_type'] = 'yolo'
    p1_yolo = Pipeline1(yolo_v5, yolo_v8, device=device, config=cfg1_yolo)
    
    # 3. Pipeline 2 (Fusion)
    cfg2 = copy.deepcopy(base_config)
    p2 = Pipeline2(yolo_v8, cnn_weights, device=device, config=cfg2)
    
    # 4. Pipeline 3 (EMA)
    cfg3_ema = copy.deepcopy(base_config)
    cfg3_ema.setdefault('pipelines', {}).setdefault('pipeline3', {})['smoother_type'] = 'ema'
    p3_ema = Pipeline3(yolo_v8, device=device, config=cfg3_ema)
    
    # 5. Pipeline 3 (Majority)
    cfg3_mj = copy.deepcopy(base_config)
    cfg3_mj.setdefault('pipelines', {}).setdefault('pipeline3', {})['smoother_type'] = 'majority'
    p3_mj = Pipeline3(yolo_v8, device=device, config=cfg3_mj)
    
    pipelines = [p1_cnn, p1_yolo, p2, p3_ema, p3_mj]
    titles = [
        "P1: YOLOv5 + CNN",
        "P1: YOLOv5 + YOLOv8 Classifier",
        "P2: MediaPipe + YOLO + CNN Fusion",
        "P3: Direct YOLOv8 + EMA",
        "P3: Direct YOLOv8 + Majority Vote"
    ]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Preparation for output
    grid_w, grid_h = 900, 1000 
   
    cell_w, cell_h = 300, 500
    
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (grid_w, grid_h))
        
    print(f"Running comparison on {video_path}...")
    frame_n = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_n += 1
            
            # Process frame through each pipeline
            annotated_frames = []
            for p in pipelines:
                ann, _ = p.process_frame(frame)
                annotated_frames.append(ann)
            
            # Create comparison grid (2 rows, 3 columns)
            # Fill the 6th slot with a summary or empty
            # Actually, let's just use 5. 
            if len(annotated_frames) < 6:
                black_frame = np.zeros_like(annotated_frames[0])
                cv2.putText(black_frame, "BSBS MEOW", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                annotated_frames.append(black_frame)
                titles.append("Info")

            grid = create_grid(annotated_frames, titles, grid_size=(2, 3), cell_size=(cell_w, cell_h))
            
            if writer:
                writer.write(grid)
                
            if show:
                cv2.imshow("Seatbelt Pipeline Comparison", grid)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            if frame_n % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_n / elapsed
                print(f"Processed {frame_n} frames | Grid FPS: {fps:.2f}")
                
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Comparison complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seatbelt Detection Pipeline Comparison")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--yolo_v5", type=str, default="weights/v1/best_github.pt", help="Path to YOLOv5 weights")
    parser.add_argument("--yolo_v8", type=str, default="weights/v1/best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--cnn", type=str, default="weights/v1/patch_cnn.pt", help="Path to CNN weights")
    parser.add_argument("--config", type=str, default="configs/seatbelt.yaml", help="Path to YAML config")
    parser.add_argument("--output", type=str, help="Path to save comparison video")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    
    args = parser.parse_args()
    
    run_comparison(args.video, args.yolo_v5, args.yolo_v8, args.cnn, args.config, args.output, args.show)
