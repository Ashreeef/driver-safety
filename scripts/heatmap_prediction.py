"""
heatmap_prediction.py
─────────────────────
Runs ONE seatbelt-detection pipeline on a video and produces a real-time
spatial activation heatmap that shows WHICH region of the frame triggered
a seatbelt prediction.

Layout (side-by-side):
  ┌──────────────────┬──────────────────┬──────────────────┐
  │  Original Frame  │ Pipeline Output  │ Activation Map   │
  └──────────────────┴──────────────────┴──────────────────┘

Usage examples
--------------
  # Interactive preview – Pipeline 2
  python scripts/heatmap_prediction.py \
      --video test_input/seat_on.mp4 \
      --pipeline 2 \
      --yolo_v8 weights/v1/best.pt \
      --cnn weights/v1/patch_cnn.pt \
      --show

  # Save to file – Pipeline 1 (YOLO classifier variant)
  python scripts/heatmap_prediction.py \
      --video test_input/seat_on.mp4 \
      --pipeline 1 \
      --classifier_type yolo \
      --yolo_v5 weights/v1/best_github.pt \
      --yolo_v8 weights/v1/best.pt \
      --output test_input/heatmap_out.mp4 \
      --show

  # Pipeline 3 – save only
  python scripts/heatmap_prediction.py \
      --video test_input/seat_on.mp4 \
      --pipeline 3 \
      --yolo_v8 weights/v1/best.pt \
      --output test_input/heatmap_p3.mp4
"""

import cv2
import numpy as np
import argparse
import sys
import os
import yaml
import copy
import time
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.compliance.seatbelt.pipelines import Pipeline1, Pipeline2, Pipeline3

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORMAP = cv2.COLORMAP_JET   # swap to COLORMAP_INFERNO / COLORMAP_HOT if preferred
HEATMAP_ALPHA = 0.55          # blend weight of heatmap overlay on the original

def load_config(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def accumulate(heatmap: np.ndarray, x1: int, y1: int, x2: int, y2: int,
               weight: float = 1.0, local_hm: np.ndarray = None) -> None:
    """Add *weight* to pixels inside the bounding box, optionally guided by Grad-CAM."""
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(heatmap.shape[1] - 1, x2)
    y2 = min(heatmap.shape[0] - 1, y2)
    
    if x2 > x1 and y2 > y1:
        if local_hm is not None:
            # Resize local Grad-CAM heatmap to fit the bbox
            h, w = y2 - y1, x2 - x1
            local_resized = cv2.resize(local_hm, (w, h))
            heatmap[y1:y2, x1:x2] += local_resized * weight
        else:
            # Fallback to uniform box accumulation
            heatmap[y1:y2, x1:x2] += weight


def render_heatmap(raw_frame: np.ndarray, heatmap: np.ndarray,
                   frame_count: int) -> np.ndarray:
    """
    Normalise the accumulated heatmap float array → 0-255 uint8,
    apply a jet colormap, and alpha-blend onto the original frame.
    Also draws a colour-bar legend on the right edge.
    """
    H, W = raw_frame.shape[:2]

    if heatmap.max() < 1e-6:
        # Nothing accumulated yet – return grey placeholder
        out = raw_frame.copy()
        cv2.putText(out, "No 'ON' predictions yet", (20, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return out

    # Gaussian blur to soften edges - REDUCED for Grad-CAM detail
    k = max(3, (min(H, W) // 60) | 1)   # changed from //20 to //60
    blurred = cv2.GaussianBlur(heatmap, (k, k), 0)

    normed = cv2.normalize(blurred, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normed, COLORMAP)

    # Blend
    base = raw_frame.copy()
    mask = normed > 0  # only colour pixels that actually accumulated heat
    overlay = base.copy()
    overlay[mask] = cv2.addWeighted(base, 1.0 - HEATMAP_ALPHA,
                                    colored, HEATMAP_ALPHA, 0)[mask]

    # ── colour-bar (right 20px) ──────────────────────────────────────────────
    bar_w = 20
    bar = np.linspace(255, 0, H, dtype=np.uint8).reshape(H, 1)
    bar_rgb = cv2.applyColorMap(
        np.repeat(bar, bar_w, axis=1), COLORMAP)
    overlay[:, W - bar_w:] = bar_rgb
    # Labels
    cv2.putText(overlay, "HIGH", (W - bar_w - 2, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(overlay, "LOW",  (W - bar_w - 2, H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # ── frame counter ────────────────────────────────────────────────────────
    cv2.putText(overlay, f"Frames accumulated: {frame_count}",
                (6, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1)

    return overlay


def build_strip(original: np.ndarray, annotated: np.ndarray,
                heatmap_vis: np.ndarray, cell_w: int, cell_h: int,
                pipeline_name: str) -> np.ndarray:
    """
    Resize all three panels to (cell_w × cell_h) and stack horizontally.
    """
    def prep(img, title):
        r = cv2.resize(img, (cell_w, cell_h))
        cv2.rectangle(r, (0, 0), (cell_w, 26), (20, 20, 20), -1)
        cv2.putText(r, title, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1)
        return r

    left   = prep(original,    "Original")
    middle = prep(annotated,   pipeline_name)
    right  = prep(heatmap_vis, "Activation Heatmap")

    strip = np.concatenate([left, middle, right], axis=1)

    # Thin separator lines
    h = strip.shape[0]
    strip[:, cell_w - 1:cell_w + 1]         = 40
    strip[:, cell_w * 2 - 1:cell_w * 2 + 1] = 40

    return strip


def extract_active_regions(pipeline, result, frame) -> list:
    """
    Given the raw result returned by each pipeline's process_frame, return a
    list of (x1, y1, x2, y2, confidence, local_heatmap) for every region 
    that contributed to an ON prediction.
    """
    H, W = frame.shape[:2]
    regions = []
    
    if not hasattr(pipeline, '_grad_cams'):
        pipeline._grad_cams = {}

    def get_gradcam(roi, model_obj, key):
        if key not in pipeline._grad_cams:
            from src.utils.gradcam import YOLOGradCAM, GradCAM
            # Check for SeatbeltClassifier first (it has its own grad_cam instance)
            if hasattr(model_obj, 'get_grad_cam'):
                return model_obj.get_grad_cam(roi)
            # Check for YOLO (ultralytics.YOLO class usually has 'model' and 'predictor')
            elif hasattr(model_obj, 'predict') and hasattr(model_obj, 'model') and not hasattr(model_obj, 'grad_cam'):
                pipeline._grad_cams[key] = YOLOGradCAM(model_obj)
            else:
                return None
        
        gc = pipeline._grad_cams[key]
        if hasattr(gc, 'get_heatmap'): # YOLOGradCAM
            return gc.get_heatmap(roi)
        return None

    # ── Pipeline 1 new format: scene dict ───────────────────────────────────
    if isinstance(result, dict) and 'detections' in result:
        scene_label = result.get('label', 'OFF')
        if scene_label == 'ON':
            dets = result['detections']
            if dets:
                for det in dets:
                    if det.get('label') == 'ON':
                        x1, y1, x2, y2 = det['bbox']
                        conf = det.get('confidence', 1.0)
                        roi = frame[y1:y2, x1:x2]
                        
                        hm = None
                        if hasattr(pipeline, 'classifier'):
                            hm = get_gradcam(roi, pipeline.classifier, 'clf')
                        elif hasattr(pipeline, 'yolo_classifier'):
                            hm = get_gradcam(roi, pipeline.yolo_classifier, 'yolo_clf')
                            
                        regions.append((x1, y1, x2, y2, conf, hm))
                
                if not regions and result.get('bbox'):
                    x1, y1, x2, y2 = result['bbox']
                    conf = result.get('confidence', 1.0)
                    roi = frame[y1:y2, x1:x2]
                    regions.append((x1, y1, x2, y2, conf, get_gradcam(roi)))

    # ── Pipelines 2 & 3: single label dict ──────────────────────────────────
    elif isinstance(result, dict):
        label = result.get('label', 'OFF')
        if label == 'ON':
            conf = result.get('confidence', result.get('yolo_conf', 1.0))
            bbox = result.get('bbox')
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                roi = frame[y1:y2, x1:x2]
                
                hm = None
                if hasattr(pipeline, 'classifier'):
                    hm = get_gradcam(roi, pipeline.classifier, 'clf')
                elif hasattr(pipeline, 'yolo_classifier'):
                    hm = get_gradcam(roi, pipeline.yolo_classifier, 'yolo_clf')
                
                regions.append((x1, y1, x2, y2, conf, hm))
            else:
                # Pipeline 3 / Full frame YOLO
                hm = None
                if hasattr(pipeline, 'yolo_classifier'):
                    hm = get_gradcam(frame, pipeline.yolo_classifier, 'yolo_clf')
                regions.append((0, 0, W, H, conf, hm))

    # ── Pipeline 1 old / list format (backward compat) ──────────────────────
    elif isinstance(result, list):
        for det in result:
            if det.get('label') == 'ON':
                x1, y1, x2, y2 = det['bbox']
                roi = frame[y1:y2, x1:x2]
                regions.append((x1, y1, x2, y2, det.get('confidence', 1.0), get_gradcam(roi)))

    return regions


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(args, config: dict):
    device = 'cpu'

    if args.pipeline == 1:
        cfg = copy.deepcopy(config)
        cfg.setdefault('pipelines', {}).setdefault('pipeline1', {})\
           ['classifier_type'] = args.classifier_type
        classifier_path = (args.yolo_v8 if args.classifier_type == 'yolo'
                           else args.cnn)
        p = Pipeline1(args.yolo_v5, classifier_path,
                      device=device, config=cfg)
        name = f"Pipeline 1 – YOLOv5 + {args.classifier_type.upper()}"

    elif args.pipeline == 2:
        cfg = copy.deepcopy(config)
        p = Pipeline2(args.yolo_v8, args.cnn, device=device, config=cfg)
        name = "Pipeline 2 – MediaPipe + YOLO/CNN Fusion"

    elif args.pipeline == 3:
        cfg = copy.deepcopy(config)
        cfg.setdefault('pipelines', {}).setdefault('pipeline3', {})\
           ['smoother_type'] = args.smoother_type
        p = Pipeline3(args.yolo_v8, device=device, config=cfg)
        name = f"Pipeline 3 – YOLOv8n + {args.smoother_type.upper()}"

    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}. Choose 1, 2, or 3.")

    return p, name


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"  Seatbelt Prediction Heatmap Visualiser")
    print(f"{'='*60}")
    print(f"  Video     : {args.video}")
    print(f"  Pipeline  : {args.pipeline}")
    print(f"  Output    : {args.output or '(none – display only)'}")
    print(f"{'='*60}\n")

    print("Initialising pipeline …")
    pipeline, pipeline_name = build_pipeline(args, config)
    print(f"  ✓  {pipeline_name}\n")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    ret, first = cap.read()
    if not ret:
        sys.exit("[ERROR] Video is empty.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

    H, W = first.shape[:2]
    cell_w = args.cell_width
    cell_h = int(cell_w * H / W)          # keep aspect ratio
    strip_w = cell_w * 3
    strip_h = cell_h

    # Floating-point accumulator (same resolution as source)
    heatmap = np.zeros((H, W), dtype=np.float32)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_cap = cap.get(cv2.CAP_PROP_FPS) or 25
        writer = cv2.VideoWriter(args.output, fourcc, fps_cap,
                                 (strip_w, strip_h))
        print(f"Writing output → {args.output}  ({strip_w}×{strip_h} @ {fps_cap:.1f} fps)")

    frame_n = 0
    on_frames = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

            # ── run pipeline ────────────────────────────────────────────────
            annotated, result = pipeline.process_frame(frame)

            # ── extract active regions & accumulate ──────────────────────────
            regions = extract_active_regions(pipeline, result, frame)
            if regions:
                on_frames += 1
                for i, (x1, y1, x2, y2, conf, local_hm) in enumerate(regions):
                    if local_hm is not None:
                        if frame_n % 30 == 0:
                            print(f"  [DEBUG] Applying Grad-CAM for region {i} (conf: {conf:.2f})")
                    accumulate(heatmap, x1, y1, x2, y2, weight=float(conf), local_hm=local_hm)

            # ── render heatmap panel ────────────────────────────────────────
            heatmap_vis = render_heatmap(frame, heatmap, on_frames)

            # ── build strip ─────────────────────────────────────────────────
            strip = build_strip(frame, annotated, heatmap_vis,
                                cell_w, cell_h, pipeline_name)

            # ── stats overlay (bottom of strip) ─────────────────────────────
            elapsed = time.time() - t0
            fps = frame_n / elapsed if elapsed > 0 else 0
            pct = f"{100*on_frames/frame_n:.1f}%" if frame_n else "0%"
            info = (f"Frame {frame_n}  |  ON rate: {pct}  "
                    f"|  FPS: {fps:.1f}  |  q = quit")
            cv2.rectangle(strip, (0, strip_h - 22), (strip_w, strip_h),
                          (10, 10, 10), -1)
            cv2.putText(strip, info, (8, strip_h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if writer:
                writer.write(strip)

            if args.show:
                cv2.imshow("Seatbelt Activation Heatmap", strip)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n[User] Quit requested.")
                    break

            if frame_n % 30 == 0:
                print(f"  Frame {frame_n:5d}  |  ON: {on_frames:4d} ({pct})  "
                      f"|  FPS: {fps:.1f}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Done.  Processed {frame_n} frames in {time.time()-t0:.1f}s")
    print(f"  Seatbelt ON predictions  : {on_frames} / {frame_n} frames "
          f"({100*on_frames/max(frame_n,1):.1f}%)")
    if heatmap.max() > 0:
        peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        print(f"  Peak activation pixel    : ({peak_x}, {peak_y})  "
              f"(x={peak_x/W:.1%} across, y={peak_y/H:.1%} down)")
    if args.output:
        print(f"  Output saved to          : {args.output}")
    print(f"{'─'*60}\n")

    # ── save static heatmap PNG ──────────────────────────────────────────────
    if args.save_png:
        final_vis = render_heatmap(first, heatmap, on_frames)
        cv2.imwrite(args.save_png, final_vis)
        print(f"  Static heatmap PNG → {args.save_png}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Seatbelt Prediction Activation Heatmap",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=__doc__
    )
    # Required
    p.add_argument("--video",    required=True,
                   help="Path to input video file")
    p.add_argument("--pipeline", required=True, type=int, choices=[1, 2, 3],
                   help="Which pipeline to evaluate (1, 2, or 3)")

    # Weights
    p.add_argument("--yolo_v5",  default="weights/v1/best_github.pt",
                   help="YOLOv5 weights (Pipeline 1 only)")
    p.add_argument("--yolo_v8",  default="weights/v1/best.pt",
                   help="YOLOv8n weights  (Pipelines 2, 3)")
    p.add_argument("--cnn",      default="weights/v1/patch_cnn.pt",
                   help="CNN patch classifier weights (Pipelines 1, 2)")

    # Pipeline-specific knobs
    p.add_argument("--classifier_type", default="cnn", choices=["cnn", "yolo"],
                   help="Pipeline 1 classifier variant  [default: cnn]")
    p.add_argument("--smoother_type",   default="ema",
                   choices=["ema", "majority"],
                   help="Pipeline 3 smoother variant    [default: ema]")

    # Config / output
    p.add_argument("--config",   default="configs/seatbelt.yaml",
                   help="Path to YAML config file")
    p.add_argument("--output",   default=None,
                   help="Save heatmap video to this path (.mp4)")
    p.add_argument("--save_png", default=None,
                   help="Save final static heatmap image to this path (.png)")
    p.add_argument("--show",     action="store_true",
                   help="Show live preview window (press q to quit)")
    p.add_argument("--cell_width", type=int, default=380,
                   help="Width of each panel cell in pixels [default: 380]")

    return p.parse_args()


if __name__ == "__main__":
    os.chdir(ROOT)   # make relative weight/config paths work
    run(parse_args())
