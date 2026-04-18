"""
unified_inference.py
=========================
Production-ready unified inference pipeline.
Integrates Smoking Detection and Seatbelt Detection (Pipeline 2).
"""

import argparse
import os
import sys
import time
import yaml
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# ── Add project root to path ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Modular Imports ─────────────────────────────────────────────────────────
from src.compliance.smoking.landmarks import LandmarkExtractor
from src.compliance.smoking.detector import SmokingDetector, SmokingFrameResult
from src.compliance.seatbelt.pipelines import Pipeline1

# ══════════════════════════════════════════════════════════════════════════════
# §1  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_yaml(path: str) -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ══════════════════════════════════════════════════════════════════════════════
# §2  UNIFIED OVERLAY RENDERER
# ══════════════════════════════════════════════════════════════════════════════

class OverlayRenderer:
    """
    Renders a unified HUD for both Smoking and Seatbelt status.
    Improved for clarity and professional look.
    """
    def __init__(self, smoking_cfg: Dict, seatbelt_cfg: Dict):
        self.sm_cfg = smoking_cfg.get('visualization', {})
        self.sb_cfg = seatbelt_cfg.get('visualization', {})
        self.sm_colors = self.sm_cfg.get('colors', {})
        self.sb_colors = self.sb_cfg.get('colors', {})

    def _filled_rect_alpha(self, frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple, alpha: float):
        ov = frame.copy()
        cv2.rectangle(ov, pt1, pt2, color, -1)
        cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0, frame)

    def _put(self, frame: np.ndarray, text: str, pos: Tuple[int, int], scale: float, color: Tuple, thickness: int = 2):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    def render(self, frame: np.ndarray, sm_res: SmokingFrameResult, sb_res: Dict, frame_idx: int) -> np.ndarray:
        out = frame.copy()
        H, W = out.shape[:2]
        
        ph = 90 # top panel height
        
        # Top Panel (Unified)
        self._filled_rect_alpha(out, (0, 0), (W, ph), (15, 15, 15), 0.85)

        # ── SEATBELT Segment ───────────────────────────────────────────────
        sb_label = sb_res.get('label', 'OFF')
        sb_conf = sb_res.get('confidence', 0.0)
        # Use colors from seatbelt config mapping
        sb_color_list = self.sb_colors.get(sb_label, [160, 160, 160])
        if sb_label == 'ON' and True in self.sb_colors: sb_color_list = self.sb_colors[True]
        if sb_label == 'OFF' and False in self.sb_colors: sb_color_list = self.sb_colors[False]
        sb_color = tuple(sb_color_list)
        
        self._put(out, "■ SEATBELT", (20, 30), 0.5, (180, 180, 180), 1)
        self._put(out, f"{sb_label}  {sb_conf:.0%}", (20, 65), 0.9, sb_color, 2)

        # ── SMOKING Segment ────────────────────────────────────────────────
        mid = W // 2
        sm_color = tuple(self.sm_colors.get('alert' if sm_res.alert_active else 'normal', [0, 200, 0]))
        sm_label = "⚠ SMOKING" if sm_res.alert_active else "✓ CLEAR"
        sm_detail = f"FU:{sm_res.fusion_score:.2f}  TC:{sm_res.temporal_conf:.2f}"
        
        self._put(out, "■ SMOKING", (mid + 20, 30), 0.5, (180, 180, 180), 1)
        self._put(out, sm_label, (mid + 20, 65), 0.9, sm_color, 2)
        self._put(out, sm_detail, (mid + 20, 82), 0.4, (160, 160, 160), 1)

        # ── Alert Banners (Central) ────────────────────────────────────────
        alerts = []
        # In Pipeline2, label 'OFF' means seatbelt is not detected or predicted off.
        if sb_label == 'OFF': 
            alerts.append("SEATBELT NOT FASTENED")
        if sm_res.alert_active: 
            alerts.append("SMOKING DETECTED")
        
        if alerts:
            # Banner height/color based on severity
            banner_y = ph
            msg = " & ".join(alerts)
            b_color = (0, 0, 180) if len(alerts) > 1 else (0, 100, 200)
            self._filled_rect_alpha(out, (0, banner_y), (W, banner_y + 35), b_color, 0.8)
            self._put(out, "⚠  ALERT: " + msg, (15, banner_y + 25), 0.65, (255, 255, 255), 2)

        # ── Visual Indicators (ROI/Landmarks) ───────────────────────────────
        # Seatbelt ROI/BBox
        sb_bbox = sb_res.get('bbox')
        if sb_bbox:
            x1, y1, x2, y2 = sb_bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), sb_color, 2)

        # Smoking Landmarks
        lm = sm_res.landmarks or {}
        lm_color = tuple(self.sm_colors.get('landmark', [0, 255, 130]))
        if "right_wrist" in lm:
            wx, wy = int(lm["right_wrist"][0] * W), int(lm["right_wrist"][1] * H)
            cv2.circle(out, (wx, wy), 8, lm_color, -1)
        if "mouth_centre" in lm:
            mx, my = int(lm["mouth_centre"][0] * W), int(lm["mouth_centre"][1] * H)
            cv2.circle(out, (mx, my), 5, (0, 200, 255), -1)

        # Smoking Cigarette Boxes
        det_color = tuple(self.sm_colors.get('det_box', [0, 200, 255]))
        for name, conf, bbox in sm_res.detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), det_color, 2)
            self._put(out, f"{name} {conf:.0%}", (x1, y1 - 10), 0.5, det_color, 1)

        # Bottom Metadata Bar
        self._filled_rect_alpha(out, (0, H - 30), (W, H), (10, 10, 10), 0.7)
        footer = f"F: {frame_idx} | SM_FU: {sm_res.fusion_score:.2f} | SB_RG: {sb_res.get('rate_info',{}).get('on_rate',0.0):.2%}"
        self._put(out, footer, (10, H - 10), 0.4, (150, 150, 150), 1)

        return out

# ══════════════════════════════════════════════════════════════════════════════
# §3  MAIN MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedSafetyMonitor:
    def __init__(self, smoking_cfg_path: str, seatbelt_cfg: Dict, sb_yolo: str, sb_clf: str, sm_weights: Optional[str] = None):
        self.smoking_cfg = load_yaml(smoking_cfg_path)
        self.seatbelt_cfg = seatbelt_cfg
        
        print("  Loading Smoking Subsystem...")
        self.extractor = LandmarkExtractor(self.smoking_cfg.get('landmarks', {}))
        self.smoking_detector = SmokingDetector(self.smoking_cfg, self.extractor, sm_weights)
        
        print(f"  Loading Seatbelt Subsystem (Pipeline 1: YOLO ROI)...")
        self.seatbelt_detector = Pipeline1(sb_yolo, sb_clf, config=self.seatbelt_cfg)
        
        self.renderer = OverlayRenderer(self.smoking_cfg, self.seatbelt_cfg)

    def run(self, source: str | int, output: Optional[str] = None, show: bool = False):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open source {source}")
            return

        W, H = int(cap.get(3)), int(cap.get(4))
        FPS = cap.get(5) or 30
        
        writer = None
        if output:
            writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
            print(f"💾 Recording to {output}")

        frame_idx = 0
        t_start = time.time()

        # Window resizing logic
        win_name = "Unified Safety Monitor"
        display_w = 1280
        display_h = int(display_w * (H / W))

        try:
            if show:
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_name, display_w, display_h)

            while True:
                ret, frame = cap.read()
                if not ret: break

                # Smoking Process
                sm_res = self.smoking_detector.process(frame)
                
                # Seatbelt Process (Pipeline2)
                _, sb_res = self.seatbelt_detector.process_frame(frame)
                
                # Annotate
                annotated = self.renderer.render(frame, sm_res, sb_res, frame_idx)

                if writer: writer.write(annotated)
                
                if show:
                    cv2.imshow(win_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                
                frame_idx += 1
                if frame_idx % 30 == 0:
                    fps = frame_idx / (time.time() - t_start)
                    print(f"  Processed {frame_idx} frames | FPS: {fps:.1f}")

        finally:
            cap.release()
            if writer: writer.release()
            cv2.destroyAllWindows()
            print(f"✅ Finished. Total frames: {frame_idx}")

# ══════════════════════════════════════════════════════════════════════════════
# §4  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path or camera ID")
    parser.add_argument("--sb_yolo", type=str, default="weights/v1/best_github.pt", help="Seatbelt YOLO weights")
    parser.add_argument("--sb_clf", type=str, default="weights/v1/best.pt", help="Seatbelt ROI Classifier weights")
    parser.add_argument("--sm_weights", type=str, default="weights/smoking/best.onnx", help="Smoking YOLO weights")
    parser.add_argument("--output", type=str, default=None, help="Output path to save video")
    parser.add_argument("--show", action="store_true", help="Display results in a resized window")
    
    # Rate Gate override
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rate-gate", action="store_true", dest="rate_gate", default=None, help="Enable seatbelt rate gate (overrides YAML)")
    group.add_argument("--no-rate-gate", action="store_false", dest="rate_gate", help="Disable seatbelt rate gate (overrides YAML)")
    
    args = parser.parse_args()

    # Paths
    sm_cfg = os.path.join(_PROJECT_ROOT, "configs", "smoking.yaml")
    sb_cfg_path = os.path.join(_PROJECT_ROOT, "configs", "seatbelt.yaml")
    sb_cfg = load_yaml(sb_cfg_path)
    
    # Apply CLI override for rate gate
    if args.rate_gate is not None:
        if 'rate_gate' not in sb_cfg: sb_cfg['rate_gate'] = {}
        sb_cfg['rate_gate']['enabled'] = args.rate_gate

    # Validation
    if not os.path.exists(args.sb_yolo):
        print(f"⚠ Warning: Seatbelt YOLO weights not found at {args.sb_yolo}")
    if not os.path.exists(args.sb_clf):
        print(f"⚠ Warning: Seatbelt Classifier weights not found at {args.sb_clf}")

    monitor = UnifiedSafetyMonitor(sm_cfg, sb_cfg, args.sb_yolo, args.sb_clf, args.sm_weights)
    monitor.run(args.video, args.output, args.show)
