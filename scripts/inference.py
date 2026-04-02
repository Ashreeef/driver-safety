import cv2
import argparse
import time
import os
import sys
import warnings
from pathlib import Path

# Suppress non-critical warnings from dependencies (YOLOv5/Torch)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF/MediaPipe noise

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.seatbelt.pipelines import PipelineA, PipelineB

def run_inference(pipeline_type, video_path, yolo_weights, cnn_weights, output_path=None, show=False):
    device = 'cpu' # default
    
    if pipeline_type == 'A':
        pipeline = PipelineA(yolo_weights, cnn_weights, device=device)
    else:
        pipeline = PipelineB(yolo_weights, cnn_weights, device=device)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 20
    
    writer = None
    if output_path:
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (fw, fh))
        
    print(f"Starting Pipeline {pipeline_type} on {video_path}...")
    
    frame_n = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_n += 1
            
            # Process
            annotated, results = pipeline.process_frame(frame)
            
            # Performance tracking
            if frame_n % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_n / elapsed
                print(f"Frame {frame_n} | Average FPS: {fps:.2f}")
                
            if writer:
                writer.write(annotated)
                
            if show:
                cv2.imshow(f"Pipeline {pipeline_type}", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Inference complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seatbelt Detection Inference")
    parser.add_argument("--type", type=str, choices=['A', 'B'], default='B', help="Pipeline type (A or B)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--yolo", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--cnn", type=str, required=True, help="Path to CNN weights")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    
    args = parser.parse_args()
    run_inference(args.type, args.video, args.yolo, args.cnn, args.output, args.show)
