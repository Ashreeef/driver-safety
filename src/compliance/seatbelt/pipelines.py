import cv2
import numpy as np
import os
import sys
import torch
from collections import deque
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict

from .roi import ROIExtractor, get_yolo_roi_boxes
from .classifier import SeatbeltClassifier
from .geometric import GeometricPrior
from src.fusion.smoothers import EMASmoother, MajorityVoteSmoother, BiLSTMSmoother


class BaseSeatbeltPipeline:
    """
    Interface for seatbelt detection pipelines.

    Rate-gate
    ---------
    Every pipeline inherits a sliding-window ON/OFF rate gate that fires an
    authoritative prediction whenever the rolling fraction of ON (or OFF)
    frames in the last ``window_size`` frames exceeds the configured threshold.

    Configure via seatbelt.yaml::

        rate_gate:
          enabled:       true
          window_size:   30      # frames in the rolling window
          on_threshold:  0.60    # ≥60 % ON  → force ON
          off_threshold: 0.60    # ≥60 % OFF → force OFF
    """

    # ── Rate-gate API ──────────────────────────────────────────────────────

    def _init_rate_gate(self, config: dict) -> None:
        """Initialise the sliding-window ON/OFF rate gate from *config*."""
        rg = config.get('rate_gate', {})
        self.rate_gate_enabled  = bool(rg.get('enabled',       False))
        self.rate_window_size   = int(rg.get('window_size',    30))
        self.rate_on_threshold  = float(rg.get('on_threshold',  0.60))
        self.rate_off_threshold = float(rg.get('off_threshold', 0.60))
        self._rate_history: deque = deque(maxlen=self.rate_window_size)

    def apply_rate_gate(self, label: str,
                        confidence: float) -> Tuple[str, float, Dict]:
        """
        Record the current frame label and apply the rate gate.

        Rules (only active when ``rate_gate.enabled = true``):
          • on_rate  ≥ on_threshold  → force label = ON
          • off_rate ≥ off_threshold → force label = OFF
          • Otherwise                → pass the smoother output unchanged

        Returns
        -------
        (final_label, final_confidence, rate_info)
        """
        is_on = 1 if label == 'ON' else 0
        self._rate_history.append(is_on)

        n        = len(self._rate_history)
        on_rate  = sum(self._rate_history) / n if n else 0.0
        off_rate = 1.0 - on_rate

        final_label = label
        final_conf  = confidence
        gate_fired  = False

        if self.rate_gate_enabled:
            if on_rate >= self.rate_on_threshold:
                final_label = 'ON'
                final_conf  = max(confidence, on_rate)
                gate_fired  = True
            elif off_rate >= self.rate_off_threshold:
                final_label = 'OFF'
                final_conf  = max(confidence, off_rate)
                gate_fired  = True

        return final_label, final_conf, {
            'on_rate':  on_rate,
            'off_rate': off_rate,
            'n':        n,
            'window':   self.rate_window_size,
            'fired':    gate_fired,
            'enabled':  self.rate_gate_enabled,
        }

    def _draw_rate_info(self, frame: np.ndarray,
                        rate_info: Optional[Dict]) -> None:
        """
        Draw a compact rate-gate HUD in the bottom-right corner of *frame*
        (in-place).  Skipped when *rate_info* is falsy.
        """
        if not rate_info:
            return
        H, W = frame.shape[:2]

        on_r  = rate_info.get('on_rate',  0.0)
        off_r = rate_info.get('off_rate', 0.0)
        n     = rate_info.get('n', 0)
        wind  = rate_info.get('window', 0)
        fired = rate_info.get('fired', False)
        enab  = rate_info.get('enabled', False)

        # Panel geometry (bottom-right)
        pw, ph = 158, 72
        bx = W - pw - 6
        by = H - ph - 6

        # Background
        cv2.rectangle(frame, (bx - 2, by - 2), (bx + pw, by + ph),
                      (18, 18, 18), -1)
        cv2.rectangle(frame, (bx - 2, by - 2), (bx + pw, by + ph),
                      (90, 90, 90), 1)

        # Title / status
        title_col = (0, 220, 60) if (fired and enab) else (150, 150, 150)
        status    = "FIRED" if (fired and enab) else ("IDLE" if enab else "DISABLED")
        cv2.putText(frame, f"RATE GATE  [{status}]",
                    (bx + 2, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, title_col, 1)

        # ON bar
        BAR = 86
        lbl_x, bar_x, bar_y1, bar_y2 = bx + 2, bx + 56, by + 22, by + 30
        cv2.putText(frame, f"ON  {on_r:5.1%}",
                    (lbl_x, bar_y2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + BAR, bar_y2),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y1),
                      (bar_x + int(BAR * on_r), bar_y2),
                      (0, 200, 60), -1)

        # OFF bar
        bar_y1 += 18; bar_y2 += 18
        cv2.putText(frame, f"OFF {off_r:5.1%}",
                    (lbl_x, bar_y2 - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + BAR, bar_y2),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y1),
                      (bar_x + int(BAR * off_r), bar_y2),
                      (0, 60, 220), -1)

        # Window counter
        cv2.putText(frame, f"window  {n} / {wind} frames",
                    (bx + 2, by + ph - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1)

    # ── Abstract interface ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────

class Pipeline1(BaseSeatbeltPipeline):
    """
    Pipeline 1: YOLOv5 Full-Frame ROI + CNN/YOLOv8 classifier + Majority Vote.

    Returns
    -------
    (annotated_frame, scene_dict)

    scene_dict keys
    ~~~~~~~~~~~~~~~
    detections   – list of per-box dicts  (bbox, label, confidence, …)
    label        – scene-level ON/OFF after rate gate
    confidence   – scene-level confidence
    bbox         – dominant bbox (highest-conf ON box, or first box)
    rate_info    – rate-gate diagnostics
    """

    def __init__(self, yolo_path: str, classifier_path: str,
                 device: str = 'cpu', config: Optional[Dict] = None):
        self.config   = config or {}
        self.device   = device

        roi_cfg  = self.config.get('roi', {})
        clf_cfg  = self.config.get('classifier', {})
        sm_cfg   = self.config.get('smoothers', {}).get('majority', {})
        self.viz_cfg = self.config.get('visualization', {})

        # ── YOLO ROI model ─────────────────────────────────────────────────
        try:
            self.yolo = YOLO(yolo_path)
            self.yolo(np.zeros((16, 16, 3), dtype=np.uint8), verbose=False)
            if not hasattr(self.yolo, 'predict') and not hasattr(self.yolo, 'model'):
                raise ValueError("Modern loader failed to initialize model")
        except Exception:
            import torch.hub
            hub_dir = torch.hub.get_dir()
            v5_path = os.path.join(hub_dir, "ultralytics_yolov5_master")
            if not os.path.exists(v5_path):
                torch.hub.help("ultralytics/yolov5", "custom")
            if v5_path not in sys.path:
                sys.path.insert(0, v5_path)
            self.yolo = torch.hub.load(
                "ultralytics/yolov5", "custom", path=yolo_path,
                force_reload=False)
            if hasattr(self.yolo, 'eval'):
                self.yolo.eval()

        pipeline1_cfg = self.config.get('pipelines', {}).get('pipeline1', {})
        self.classifier_type  = pipeline1_cfg.get('classifier_type', 'cnn')
        self.smoother         = MajorityVoteSmoother(config=sm_cfg)
        self.yolo_conf_thresh = roi_cfg.get('yolo_conf_thresh', 0.5)
        self.patch_conf_thresh = clf_cfg.get('patch_conf_threshold', 0.65)

        if self.classifier_type == 'cnn':
            self.classifier = SeatbeltClassifier(
                classifier_path, device=device, config=clf_cfg)
        else:
            self.yolo_classifier = YOLO(classifier_path)

        self._init_rate_gate(self.config)

    # ── process_frame ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        raw_results = []
        boxes = get_yolo_roi_boxes(
            self.yolo, frame, conf_thresh=self.yolo_conf_thresh)

        for (x1, y1, x2, y2, yolo_conf) in boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if self.classifier_type == 'yolo':
                res2 = self.yolo_classifier(roi, verbose=False)
                patch_cls, patch_conf = 0, 0.0
                for r in res2:
                    if r.boxes:
                        for b in r.boxes:
                            c = b.conf[0].item()
                            if c > patch_conf:
                                patch_conf = c
                                patch_cls  = int(b.cls[0].item())
            else:
                patch_cls, patch_conf = self.classifier.predict(roi)
                if patch_cls == 1 and patch_conf < self.patch_conf_thresh:
                    patch_cls = 0

            sm_cls, sm_conf = self.smoother.update(patch_cls, patch_conf)
            raw_results.append({
                'bbox':      (x1, y1, x2, y2),
                'yolo_conf': yolo_conf,
                'class_idx': sm_cls,
                'confidence': sm_conf,
                'label':     'ON' if sm_cls == 1 else 'OFF',
            })

        # ── Scene label (any box ON → scene ON) ───────────────────────────
        raw_label = 'ON' if any(d['label'] == 'ON' for d in raw_results) else 'OFF'
        raw_conf  = max((d['confidence'] for d in raw_results), default=0.0)

        scene_label, scene_conf, rate_info = self.apply_rate_gate(
            raw_label, raw_conf)

        # Dominant bbox: highest-conf ON box, else first box
        dominant_bbox = None
        if raw_results:
            on_boxes = [d for d in raw_results if d['label'] == 'ON']
            dominant_bbox = (on_boxes[0]['bbox'] if on_boxes
                             else raw_results[0]['bbox'])

        scene = {
            'detections': raw_results,
            'label':      scene_label,
            'confidence': scene_conf,
            'bbox':       dominant_bbox,
            'rate_info':  rate_info,
        }

        annotated = self.annotate(frame, raw_results, rate_info)
        return annotated, scene

    # ── annotate ───────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, detections: List[Dict],
                 rate_info: Optional[Dict] = None) -> np.ndarray:
        out = frame.copy()
        H, W = frame.shape[:2]

        colors_cfg = self.viz_cfg.get(
            'colors', {'ON': (0, 200, 0), 'OFF': (0, 0, 220),
                       'WARMING': (160, 160, 160)})
        colors = {}
        for k, v in colors_cfg.items():
            if k is True:   colors['ON']     = tuple(v)
            elif k is False: colors['OFF']    = tuple(v)
            else:            colors[str(k)]   = tuple(v)

        overlay_alpha = self.viz_cfg.get('overlay_alpha', 0.6)
        banner_h      = self.viz_cfg.get('banner_height', 50)

        if not detections:
            ov = out.copy()
            cv2.rectangle(ov, (0, 0), (W, banner_h), (80, 80, 80), -1)
            cv2.addWeighted(ov, overlay_alpha, out, 1.0 - overlay_alpha, 0, out)
            cv2.putText(out, "No seatbelt detected",
                        (10, int(banner_h * 0.68)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 200), 2)
        else:
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                color = colors['ON'] if d['class_idx'] == 1 else colors['OFF']
                lbl   = f"Seatbelt {d['label']} ({d['confidence']:.0%})"

                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                banner_y1 = max(0, y1 - 36)
                ov = out.copy()
                cv2.rectangle(ov, (x1, banner_y1), (x2, y1), color, -1)
                cv2.addWeighted(ov, 0.70, out, 0.30, 0, out)
                cv2.putText(out, lbl, (x1 + 5, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2)
                cv2.putText(out, f"YOLO: {d['yolo_conf']:.0%}",
                            (x1 + 5, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        cv2.putText(out, "Pipeline 1: YOLOv5 ROI + CNN",
                    (8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        self._draw_rate_info(out, rate_info)
        return out


# ──────────────────────────────────────────────────────────────────────────────

class Pipeline2(BaseSeatbeltPipeline):
    """
    Pipeline 2: MediaPipe Pose + YOLOv8 + CNN + RANSAC + EMA Fusion.

    The rate gate is applied as Stage 6, after the EMA smoother.
    """

    def __init__(self, yolo_path: str, classifier_path: str,
                 device: str = 'cpu', config: Optional[Dict] = None):
        self.config = config or {}
        self.device = device

        roi_cfg  = self.config.get('roi', {})
        clf_cfg  = self.config.get('classifier', {})
        geo_cfg  = self.config.get('geometric', {})
        sm_cfg   = self.config.get('smoothers', {}).get('ema', {})
        fus_cfg  = self.config.get('fusion', {})
        self.viz_cfg = self.config.get('visualization', {})

        self.roi_extractor = ROIExtractor(method='mediapipe', config=roi_cfg)
        self.yolo          = YOLO(yolo_path)
        self.classifier    = SeatbeltClassifier(
            classifier_path, device=device, config=clf_cfg)
        self.geo_prior     = GeometricPrior(config=geo_cfg)
        self.smoother      = EMASmoother(config=sm_cfg)

        self.fusion_cnn_weight    = fus_cfg.get('cnn_weight',          0.6)
        self.fusion_yolo_weight   = fus_cfg.get('yolo_weight',         0.4)
        self.agreement_threshold  = fus_cfg.get('agreement_threshold', 0.75)
        self.use_geo_prior        = fus_cfg.get('use_geo_prior',       False)

        self._init_rate_gate(self.config)

    # ── process_frame ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        H, W = frame.shape[:2]

        # Stage 1: ROI
        roi, bbox, kps = self.roi_extractor.extract(frame)
        if roi is None:
            roi, bbox, kps = frame.copy(), (0, 0, W, H), []

        # Stage 2: YOLO on ROI
        try:
            results = self.yolo(roi, verbose=False)
        except TypeError:
            results = self.yolo(roi)

        yolo_pred, yolo_conf = 0, 0.0
        for r in results:
            if r.boxes:
                for b in r.boxes:
                    c = b.conf[0].item()
                    if c > yolo_conf:
                        yolo_conf = c
                        yolo_pred = int(b.cls[0].item())

        # Stage 3: CNN on ROI
        cnn_pred, cnn_conf = self.classifier.predict(roi)

        # Stage 4: Geometric Prior
        prior_score = 0.5
        if self.use_geo_prior:
            both_agree = (yolo_pred == cnn_pred) and (
                max(yolo_conf, cnn_conf) > 0.75)
            if both_agree:
                prior_score = 1.0 if yolo_pred == 1 else 0.0
            else:
                prior_score = self.geo_prior.get_score(roi, kps, bbox)

        # Stage 5: EMA Fusion
        yolo_on_conf = (yolo_conf if yolo_pred == 1 else (1.0 - yolo_conf)
                        ) if yolo_conf > 0 else 0.0
        on_prob = (self.fusion_cnn_weight  * cnn_conf +
                   self.fusion_yolo_weight * yolo_on_conf)
        label, final_conf = self.smoother.update(on_prob)

        # Stage 6: Rate Gate
        label, final_conf, rate_info = self.apply_rate_gate(label, final_conf)

        res = {
            'label':      label,
            'confidence': final_conf,
            'bbox':       bbox,
            'kps':        kps,
            'yolo':       (yolo_pred, yolo_conf),
            'cnn':        (cnn_pred,  cnn_conf),
            'prior':      prior_score,
            'rate_info':  rate_info,
        }

        annotated = self.annotate(frame, res)
        return annotated, res

    # ── annotate ───────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, res: Dict) -> np.ndarray:
        out = frame.copy()
        H, W = out.shape[:2]
        label, conf = res['label'], res['confidence']
        bbox, kps   = res['bbox'],  res['kps']
        conf = conf if label == 'ON' else (1.0 - conf)

        colors_cfg = self.viz_cfg.get(
            'colors', {'ON': (0, 200, 0), 'OFF': (0, 0, 220),
                       'WARMING': (160, 160, 160)})
        colors = {}
        for k, v in colors_cfg.items():
            if k is True:   colors['ON']  = tuple(v)
            elif k is False: colors['OFF'] = tuple(v)
            else:            colors[str(k)] = tuple(v)

        color         = colors.get(label, colors.get('WARMING', (160, 160, 160)))
        overlay_alpha = self.viz_cfg.get('overlay_alpha', 0.6)
        banner_h      = self.viz_cfg.get('banner_height', 50)
        font_s        = self.viz_cfg.get('font_scale', 0.9)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        for kp in kps:
            cv2.circle(out, kp, 5, (255, 200, 0), -1)

        # Top banner
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (W, banner_h), color, -1)
        cv2.addWeighted(ov, overlay_alpha, out, 1.0 - overlay_alpha, 0, out)
        cv2.putText(out, f"Seatbelt {label}  ({conf:.0%})",
                    (10, int(banner_h * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, font_s, (255, 255, 255), 2)

        # Score cards
        models = [
            ("YOLO", "ON" if res['yolo'][0] == 1 else "OFF", res['yolo'][1]),
            ("CNN",  "ON" if res['cnn'][0]  == 1 else "OFF", res['cnn'][1]),
            ("GEO",  "ON" if res['prior']   >= 0.5 else "OFF", res['prior']),
            ("EMA",  label if label != "WARMING" else "--",    conf),
        ]
        cell_w, cell_h = 100, 56
        px, py = 8, H - cell_h - 10

        for i, (name, pred, score) in enumerate(models):
            cx, cy      = px + i * (cell_w + 6), py
            cell_color  = (colors['ON'] if pred == "ON"
                           else (colors['OFF'] if pred == "OFF"
                                 else colors.get('WARMING', (160, 160, 160))))
            ov = out.copy()
            cv2.rectangle(ov, (cx, cy), (cx + cell_w, cy + cell_h),
                          cell_color, -1)
            cv2.addWeighted(ov, overlay_alpha - 0.05, out,
                            1.05 - overlay_alpha, 0, out)
            cv2.rectangle(out, (cx, cy), (cx + cell_w, cy + cell_h),
                          (210, 210, 210), 1)
            cv2.putText(out, name, (cx + 6, cy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
            cv2.putText(out, pred, (cx + 6, cy + 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2)
            bmax = cell_w - 8
            blen = int(bmax * min(max(score, 0.0), 1.0))
            by2  = cy + cell_h - 7
            cv2.rectangle(out, (cx + 4, by2), (cx + 4 + bmax, by2 + 4),
                          (50, 50, 50), -1)
            cv2.rectangle(out, (cx + 4, by2), (cx + 4 + blen, by2 + 4),
                          (255, 255, 255), -1)

        self._draw_rate_info(out, res.get('rate_info'))
        return out


# ──────────────────────────────────────────────────────────────────────────────

class Pipeline3(BaseSeatbeltPipeline):
    """
    Pipeline 3: YOLOv8n Direct Prediction + Smoother.

    The rate gate is applied after the chosen smoother.
    """

    def __init__(self, classifier_path: str, device: str = 'cpu',
                 config: Optional[Dict] = None):
        self.config = config or {}
        self.device = device

        pipeline3_cfg     = self.config.get('pipelines', {}).get('pipeline3', {})
        self.smoother_type = pipeline3_cfg.get('smoother_type', 'ema')

        sm_cfg       = self.config.get('smoothers', {}).get(self.smoother_type, {})
        self.viz_cfg = self.config.get('visualization', {})

        self.yolo_classifier = YOLO(classifier_path)

        if self.smoother_type == 'ema':
            self.smoother = EMASmoother(config=sm_cfg)
        else:
            self.smoother = MajorityVoteSmoother(config=sm_cfg)

        self._init_rate_gate(self.config)

    # ── process_frame ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        try:
            results = self.yolo_classifier(frame, verbose=False)
        except TypeError:
            results = self.yolo_classifier(frame)

        yolo_pred, yolo_conf = 0, 0.0
        for r in results:
            if r.boxes:
                for b in r.boxes:
                    c = b.conf[0].item()
                    if c > yolo_conf:
                        yolo_conf = c
                        yolo_pred = int(b.cls[0].item())

        if self.smoother_type == 'ema':
            on_prob = yolo_conf if yolo_pred == 1 else (1.0 - yolo_conf)
            label, final_conf = self.smoother.update(on_prob)
        else:
            label_idx, final_conf = self.smoother.update(yolo_pred, yolo_conf)
            label = 'ON' if label_idx == 1 else 'OFF'

        # Rate Gate
        label, final_conf, rate_info = self.apply_rate_gate(label, final_conf)

        res = {
            'label':      label,
            'confidence': final_conf,
            'yolo_conf':  yolo_conf,
            'yolo_pred':  yolo_pred,
            'rate_info':  rate_info,
        }

        annotated = self.annotate(frame, res)
        return annotated, res

    # ── annotate ───────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, res: Dict) -> np.ndarray:
        out = frame.copy()
        H, W = frame.shape[:2]

        colors_cfg = self.viz_cfg.get(
            'colors', {'ON': (0, 200, 0), 'OFF': (0, 0, 220),
                       'WARMING': (160, 160, 160)})
        colors = {}
        for k, v in colors_cfg.items():
            if k is True:   colors['ON']  = tuple(v)
            elif k is False: colors['OFF'] = tuple(v)
            else:            colors[str(k)] = tuple(v)

        label = res['label']
        conf  = res['confidence']
        color = colors.get(label, colors.get('WARMING', (160, 160, 160)))

        banner_h      = self.viz_cfg.get('banner_height', 50)
        overlay_alpha = self.viz_cfg.get('overlay_alpha', 0.6)

        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (W, banner_h), color, -1)
        cv2.addWeighted(ov, overlay_alpha, out, 1.0 - overlay_alpha, 0, out)
        cv2.putText(out, f"Seatbelt {label}  ({conf:.0%})",
                    (10, int(banner_h * 0.68)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(out,
                    f"Pipeline 3: Direct YOLOv8n + {self.smoother_type.upper()} Smoother",
                    (8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        self._draw_rate_info(out, res.get('rate_info'))
        return out
