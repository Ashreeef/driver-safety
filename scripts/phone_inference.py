import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import yaml

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.compliance.phone import PhoneDetector, VideoAnalyzer


def load_config(config_path: str) -> Dict[str, Any]:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_default_weights() -> str:
    model_paths_cfg = Path("configs") / "model_paths.yaml"
    if not model_paths_cfg.exists():
        return "weights/v1/yolov8n_phone.pt"

    with open(model_paths_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    phone_cfg = cfg.get("phone", {}) if isinstance(cfg, dict) else {}
    return str(phone_cfg.get("yolo", "weights/v1/yolov8n_phone.pt"))


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_image(detector: PhoneDetector, image_path: str, output_path: str | None) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    result = detector.detect(frame)
    print(f"Detections: {len(result.detections)} | Phones: {result.n_phones} | Alert: {result.alert_triggered}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result.annotated_frame)
        print(f"Saved annotated image: {output_path}")


def run_video(
    detector: PhoneDetector,
    video_path: str,
    output_path: str | None,
    show: bool,
    max_frames: int | None,
    frame_skip: int,
    save_analysis: bool,
    window_width: int | None,
    window_height: int | None,
) -> None:
    if not save_analysis:
        summary = detector.run_on_video(
            source=video_path,
            output_path=output_path,
            display=show,
            max_frames=max_frames,
            window_width=window_width,
            window_height=window_height,
        )
        print(json.dumps(summary, indent=2))
        return

    analyzer = VideoAnalyzer(detector)
    log = analyzer.process(
        video_path=video_path,
        output_path=output_path,
        max_frames=max_frames,
        frame_skip=frame_skip,
    )
    segments = analyzer.get_segments(log)

    phone_frames = sum(1 for r in log if r.get("alert"))
    summary = {
        "frames_processed": len(log),
        "frames_with_phone": phone_frames,
        "phone_rate": (phone_frames / len(log)) if log else 0.0,
        "events": len(segments),
    }

    analysis_dir = Path("outputs") / "phone_analysis"
    save_json(log, str(analysis_dir / "frame_log.json"))
    save_json(segments, str(analysis_dir / "events.json"))
    save_json(summary, str(analysis_dir / "summary.json"))

    print(json.dumps(summary, indent=2))
    print(f"Saved analysis artifacts in: {analysis_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phone detection inference and video analysis")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=str, help="Path to a single input image")
    source_group.add_argument("--video", type=str, help="Path to an input video")

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to phone model weights (.pt). Defaults to phone.yolo from configs/model_paths.yaml",
    )
    parser.add_argument("--config", type=str, default="configs/phone.yaml", help="Path to phone config YAML")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu, 0, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Output path for annotated media")
    parser.add_argument("--show", action="store_true", help="Show live video preview")
    parser.add_argument("--window-width", type=int, default=None, help="Initial preview window width in pixels")
    parser.add_argument("--window-height", type=int, default=None, help="Initial preview window height in pixels")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process for video")
    parser.add_argument("--frame-skip", type=int, default=1, help="Analyze every Nth frame when --save-analysis")
    parser.add_argument("--save-analysis", action="store_true", help="Save frame log and phone event JSON files")

    args = parser.parse_args()

    weights_path = args.weights or resolve_default_weights()
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"Phone weights not found: {weights_path}. "
            "Expected YOLOv8 weights at weights/v1/yolov8n_phone.pt"
        )

    cfg = load_config(args.config)
    det_cfg = cfg.get("detection", {})
    conf_threshold = args.conf if args.conf is not None else float(det_cfg.get("conf_threshold", 0.25))
    device = args.device if args.device is not None else str(det_cfg.get("device", "cpu"))
    show_driver = bool(cfg.get("visualization", {}).get("show_driver_context", True))

    detector = PhoneDetector(
        model_path=weights_path,
        conf_threshold=conf_threshold,
        device=device,
        show_driver_context=show_driver,
    )

    if args.image:
        run_image(detector, args.image, args.output)
    else:
        run_video(
            detector=detector,
            video_path=args.video,
            output_path=args.output,
            show=args.show,
            max_frames=args.max_frames,
            frame_skip=max(args.frame_skip, 1),
            save_analysis=args.save_analysis,
            window_width=args.window_width,
            window_height=args.window_height,
        )


if __name__ == "__main__":
    main()
