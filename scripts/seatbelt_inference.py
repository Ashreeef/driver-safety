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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF/MediaPipe noise

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.seatbelt.pipelines import Pipeline1, Pipeline2, Pipeline3


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_inference(pipeline_type: str, video_path: str,
                  yolo_weights: str, classifier_weights: str,
                  config: dict = None, model_paths: dict = None,
                  output_path: str = None, show: bool = False):
    """
    Run a single seatbelt pipeline synchronously on every frame of a video.

    Key difference vs run_demo
    --------------------------
    This script is **synchronous** — every frame is guaranteed to pass through
    the pipeline (and the temporal smoother) before the next frame is read.

    run_demo uses an async compliance worker that may skip frames when the
    main thread produces them faster than the worker can consume them.  This
    means the smoother accumulates a slightly sparser history in run_demo,
    which can produce small per-frame label/confidence differences on the same
    video.  The final stable prediction should be the same, but transient
    disagreements near boundary frames are expected.
    """
    config      = config or {}
    model_paths = model_paths or {}

    # ── Build pipeline ────────────────────────────────────────────────────────
    if pipeline_type == '1':
        # classifier_weights = YOLOv8n path  when classifier_type=='yolo'
        #                      MobileNetV3   when classifier_type=='cnn'
        pipeline = Pipeline1(yolo_weights, classifier_weights,
                             device='cpu', config=config)
        clf_type = config.get('pipelines', {}).get('pipeline1', {}).get(
            'classifier_type', 'cnn')
        print(f"[seatbelt_inference] Pipeline 1 loaded "
              f"(YOLOv5s ROI + {'YOLOv8n' if clf_type == 'yolo' else 'CNN'}).")

    elif pipeline_type == '2':
        # model_paths is required so Pipeline2 can find the MediaPipe pose
        # landmarker for ROI extraction.  Without it the ROI extractor falls
        # back to the full frame, giving results inconsistent with run_demo.
        pipeline = Pipeline2(yolo_weights, classifier_weights,
                             device='cpu', config=config,
                             model_paths=model_paths)
        print("[seatbelt_inference] Pipeline 2 loaded (Pose ROI + YOLOv8n + CNN).")

    elif pipeline_type == '3':
        # Pipeline 3 is a direct YOLOv8n detector.  It uses yolo_weights
        # as its classifier — classifier_weights is not used by this pipeline.
        pipeline = Pipeline3(yolo_weights, device='cpu', config=config)
        print("[seatbelt_inference] Pipeline 3 loaded (Direct YOLOv8n).")

    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 20
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pace display to real-time so every annotated frame is visible
    frame_delay_ms = max(1, int(1000 / fps_src))

    print(f"  video      : {video_path}")
    print(f"  resolution : {fw}x{fh}  |  fps: {fps_src:.1f}  |  frames: {total}")
    print(f"  delay      : {frame_delay_ms} ms/frame")

    # ── Output writer ─────────────────────────────────────────────────────────
    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (fw, fh))
        print(f"  output     : {output_path}")

    frame_n    = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_n += 1

            # ── Synchronous inference (every frame, no skipping) ──────────
            annotated, results = pipeline.process_frame(frame)

            # Periodic console status
            if frame_n % 30 == 0:
                elapsed = time.time() - start_time
                fps     = frame_n / elapsed
                print(f"  frame {frame_n:5d}/{total}  |  avg FPS: {fps:5.2f}  |  "
                      f"label={results.get('label', '?'):3s}  "
                      f"conf={results.get('confidence', 0.0):.2%}")

            if writer:
                writer.write(annotated)

            if show:
                window_name = f"Seatbelt Pipeline {pipeline_type}"
                viz_cfg = config.get('visualization', {})
                dw = viz_cfg.get('display_width',  1280)
                dh = viz_cfg.get('display_height',  720)

                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, dw, dh)
                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    print("[seatbelt_inference] Paused — press any key to resume.")
                    cv2.waitKey(0)

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"Inference complete — {frame_n} frames in {elapsed:.1f}s "
              f"({frame_n / elapsed:.1f} fps avg)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone seatbelt detection inference (synchronous, every frame).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weight mapping per pipeline type
  --type 1  --yolo <yolov5s_roi.pt>   --classifier <yolov8n.pt | mobilenetv3.pt>
  --type 2  --yolo <yolov8n.pt>       --classifier <mobilenetv3.pt>
  --type 3  --yolo <yolov8n.pt>       (--classifier is not used)
""")

    parser.add_argument("--type", type=str, choices=['1', '2', '3'], default='2',
                        help="Pipeline: 1=YOLOv5s ROI+clf, 2=Pose+YOLOv8n+CNN, 3=Direct YOLOv8n")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video file")
    parser.add_argument("--yolo", type=str, required=True,
                        help=("YOLO weights "
                              "(P1 → YOLOv5s ROI; P2/P3 → YOLOv8n)"))
    parser.add_argument("--classifier", type=str, default=None,
                        dest="classifier",
                        help=("Classifier weights for Pipeline 1: "
                              "YOLOv8n when classifier_type=yolo, "
                              "MobileNetV3 when classifier_type=cnn. "
                              "Auto-resolved from model_paths.yaml if omitted. "
                              "Ignored for Pipelines 2 and 3."))
    parser.add_argument("--config", type=str, default="configs/seatbelt.yaml",
                        help="Seatbelt YAML config (default: configs/seatbelt.yaml)")
    parser.add_argument("--model-paths", type=str,
                        default="configs/model_paths.yaml",
                        help="model_paths YAML — required for Pipeline 2 MediaPipe ROI")
    parser.add_argument("--output", type=str, default=None,
                        help="Save annotated output video to this path")
    parser.add_argument("--show", action="store_true",
                        help="Show live preview window")

    # Rate Gate override
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument("--rate-gate", action="store_true",
                            dest="rate_gate", default=None,
                            help="Enable rate gate (overrides YAML)")
    gate_group.add_argument("--no-rate-gate", action="store_false",
                            dest="rate_gate",
                            help="Disable rate gate (overrides YAML)")

    args = parser.parse_args()

    # ── Load both config files ────────────────────────────────────────────────
    cfg         = load_config(args.config)
    model_paths = load_config(args.model_paths)

    # ── Resolve classifier path when not explicitly supplied ─────────────────
    classifier_weights = args.classifier
    if classifier_weights is None:
        p1_clf_type = cfg.get('pipelines', {}).get('pipeline1', {}).get(
            'classifier_type', 'cnn')
        sb_paths = model_paths.get('seatbelt', {})
        if p1_clf_type == 'yolo':
            classifier_weights = sb_paths.get('yolo_p2', '')
            print(f"[seatbelt_inference] classifier_type=yolo → "
                  f"auto-resolved classifier to yolo_p2: {classifier_weights}")
        else:
            classifier_weights = sb_paths.get('classifier', '')
            print(f"[seatbelt_inference] classifier_type=cnn → "
                  f"auto-resolved classifier to CNN: {classifier_weights}")

    # ── Rate gate CLI override ────────────────────────────────────────────────
    if args.rate_gate is not None:
        if 'rate_gate' not in cfg:
            cfg['rate_gate'] = {}
        cfg['rate_gate']['enabled'] = args.rate_gate

    run_inference(
        pipeline_type=args.type,
        video_path=args.video,
        yolo_weights=args.yolo,
        classifier_weights=classifier_weights,
        config=cfg,
        model_paths=model_paths,
        output_path=args.output,
        show=args.show,
    )
