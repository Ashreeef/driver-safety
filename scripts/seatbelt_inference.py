import cv2
import argparse
import time
import os
import sys
import warnings
import yaml
from pathlib import Path

# Suppress non-critical warnings from dependencies (YOLOv5/Torch)
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF/MediaPipe noise

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.seatbelt.pipelines import Pipeline1, Pipeline2, Pipeline3

def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_inference(pipeline_type, video_path, yolo_weights, cnn_weights, config=None, output_path=None, show=False):
    device = 'cpu' # default
    config = config or {}
    
    if pipeline_type == '1':
        pipeline = Pipeline1(yolo_weights, cnn_weights, device=device, config=config)
    elif pipeline_type == '2':
        pipeline = Pipeline2(yolo_weights, cnn_weights, device=device, config=config)
    elif pipeline_type == '3':
        pipeline = Pipeline3(cnn_weights, device=device, config=config)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
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
                window_name = f"Pipeline {pipeline_type}"
                # Get display size from config
                viz_cfg = config.get('visualization', {})
                dw = viz_cfg.get('display_width', 1280)
                dh = viz_cfg.get('display_height', 720)
                
                # Create resizable window
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, dw, dh)
                
                cv2.imshow(window_name, annotated)
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
    parser.add_argument("--type", type=str, choices=['1', '2', '3'], default='2', help="Pipeline type (1, 2, or 3)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--yolo", type=str, required=True, help="Path to YOLO weights")
    parser.add_argument("--cnn", type=str, required=True, help="Path to CNN weights")
    parser.add_argument("--config", type=str, default="configs/seatbelt.yaml", help="Path to YAML config")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    
    # Rate Gate override
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rate-gate", action="store_true", dest="rate_gate", default=None, help="Enable rate gate (overrides YAML)")
    group.add_argument("--no-rate-gate", action="store_false", dest="rate_gate", help="Disable rate gate (overrides YAML)")
    
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Apply CLI override for rate gate
    if args.rate_gate is not None:
        if 'rate_gate' not in cfg: cfg['rate_gate'] = {}
        cfg['rate_gate']['enabled'] = args.rate_gate
    
    run_inference(args.type, args.video, args.yolo, args.cnn, config=cfg, output_path=args.output, show=args.show)
