import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict

from .roi import ROIExtractor, get_yolo_roi_boxes
from .classifier import SeatbeltClassifier
from .geometric import GeometricPrior
from src.fusion.smoothers import EMASmoother, MajorityVoteSmoother, BiLSTMSmoother

class BaseSeatbeltPipeline:
    """Interface for seatbelt detection pipelines."""
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class PipelineA(BaseSeatbeltPipeline):
    """
    Pipeline A: YOLOv5 Full-Frame ROI + CNN + Majority Vote.
    Processes each detected seatbelt strap independently.
    """
    def __init__(self, yolo_path: str, classifier_path: str, device: str = 'cpu', config: Optional[Dict] = None):
        self.config = config or {}
        self.device = device
        
        # Load sub-configs
        roi_cfg = self.config.get('roi', {})
        clf_cfg = self.config.get('classifier', {})
        sm_cfg = self.config.get('smoothers', {}).get('majority', {})
        self.viz_cfg = self.config.get('visualization', {})
        
        # Models
        try:
            self.yolo = YOLO(yolo_path)
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
            self.yolo = torch.hub.load("ultralytics/yolov5", "custom", path=yolo_path, force_reload=False)
            if hasattr(self.yolo, 'eval'):
                self.yolo.eval()
        
        self.classifier = SeatbeltClassifier(classifier_path, device=device, config=clf_cfg)
        self.smoother = MajorityVoteSmoother(config=sm_cfg)
        self.yolo_conf_thresh = roi_cfg.get('yolo_conf_thresh', 0.5)
        self.patch_conf_thresh = clf_cfg.get('patch_conf_threshold', 0.65)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Processes frame and returns annotated frame + detection results.
        """
        results = []
        boxes = get_yolo_roi_boxes(self.yolo, frame, conf_thresh=self.yolo_conf_thresh)
        
        for (x1, y1, x2, y2, yolo_conf) in boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0: continue
            
            patch_cls, patch_conf = self.classifier.predict(roi)
            if patch_cls == 1 and patch_conf < self.patch_conf_thresh:
                patch_cls = 0   # apply confidence gate
            
            # Majority vote smoothing is harder per-box in raw video without tracking
            # In the notebook, it seems to smooth based on the list position which is fragile
            # For migration, we'll keep the notebook's logic but note its limitation
            sm_cls, sm_conf = self.smoother.update(patch_cls, patch_conf)
            
            results.append({
                'bbox': (x1, y1, x2, y2),
                'yolo_conf': yolo_conf,
                'class_idx': sm_cls,
                'confidence': sm_conf,
                'label': 'ON' if sm_cls == 1 else 'OFF'
            })
            
        annotated = self.annotate(frame, results)
        return annotated, results

    def annotate(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        # Implementation of annotate_pipeline_a from notebook
        out = frame.copy()
        H, W = frame.shape[:2]
        
        # Viz settings
        colors_cfg = self.viz_cfg.get('colors', {'ON': (0, 200, 0), 'OFF': (0, 0, 220), 'WARMING': (160, 160, 160)})
        # YAML parser may load ON/OFF as True/False (booleans)
        colors = {}
        for k, v in colors_cfg.items():
            if k is True: colors['ON'] = tuple(v)
            elif k is False: colors['OFF'] = tuple(v)
            else: colors[str(k)] = tuple(v)
        
        overlay_alpha = self.viz_cfg.get('overlay_alpha', 0.6)
        banner_h = self.viz_cfg.get('banner_height', 50)
        
        if not detections:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (W, banner_h), (80, 80, 80), -1)
            cv2.addWeighted(overlay, overlay_alpha, out, 1.0 - overlay_alpha, 0, out)
            cv2.putText(out, "No seatbelt detected", (10, int(banner_h * 0.68)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 200), 2)
        else:
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                color = colors['ON'] if d['class_idx'] == 1 else colors['OFF']
                label = f"Seatbelt {d['label']} ({d['confidence']:.0%})"
                
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                # Label banner
                banner_y1 = max(0, y1 - 36)
                ov = out.copy()
                cv2.rectangle(ov, (x1, banner_y1), (x2, y1), color, -1)
                cv2.addWeighted(ov, 0.70, out, 0.30, 0, out)
                cv2.putText(out, label, (x1 + 5, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2)
                # YOLO conf
                cv2.putText(out, f"YOLO: {d['yolo_conf']:.0%}", (x1 + 5, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                            
        cv2.putText(out, "Pipeline A: YOLOv5 ROI + CNN", (8, H - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return out

class PipelineB(BaseSeatbeltPipeline):
    """
    Pipeline B: MediaPipe Pose + YOLOv8 + CNN + RANSAC + EMA Fusion.
    """
    def __init__(self, yolo_path: str, classifier_path: str, device: str = 'cpu', config: Optional[Dict] = None):
        self.config = config or {}
        self.device = device
        
        # Load sub-configs
        roi_cfg = self.config.get('roi', {})
        clf_cfg = self.config.get('classifier', {})
        geo_cfg = self.config.get('geometric', {})
        sm_cfg = self.config.get('smoothers', {}).get('ema', {})
        fus_cfg = self.config.get('fusion', {})
        self.viz_cfg = self.config.get('visualization', {})
        
        self.roi_extractor = ROIExtractor(method='mediapipe', config=roi_cfg)
        self.yolo = YOLO(yolo_path)
        self.classifier = SeatbeltClassifier(classifier_path, device=device, config=clf_cfg)
        self.geo_prior = GeometricPrior(config=geo_cfg)
        self.smoother = EMASmoother(config=sm_cfg)
        
        self.fusion_cnn_weight = fus_cfg.get('cnn_weight', 0.6)
        self.fusion_yolo_weight = fus_cfg.get('yolo_weight', 0.4)
        self.agreement_threshold = fus_cfg.get('agreement_threshold', 0.75)
        self.use_geo_prior = fus_cfg.get('use_geo_prior', True)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        H, W = frame.shape[:2]
        
        # Stage 1: ROI
        roi, bbox, kps = self.roi_extractor.extract(frame)
        if roi is None:
            roi, bbox, kps = frame.copy(), (0, 0, W, H), []
            
        # Stage 2: YOLO on ROI
        results = self.yolo(roi, verbose=False)
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
            both_agree = (yolo_pred == cnn_pred) and (max(yolo_conf, cnn_conf) > 0.75)
            if both_agree:
                prior_score = 1.0 if yolo_pred == 1 else 0.0
            else:
                prior_score = self.geo_prior.get_score(roi, kps, bbox)
                
        # Stage 5: EMA Fusion
        if yolo_conf > 0:
            yolo_on_conf = yolo_conf if yolo_pred == 1 else (1.0 - yolo_conf)
        else:
            # If YOLO detects nothing, it doesn't support an "ON" prediction
            yolo_on_conf = 0.0
            
        on_prob = self.fusion_cnn_weight * cnn_conf + self.fusion_yolo_weight * yolo_on_conf
        label, final_conf = self.smoother.update(on_prob)
        
        res = {
            'label': label,
            'confidence': final_conf,
            'bbox': bbox,
            'kps': kps,
            'yolo': (yolo_pred, yolo_conf),
            'cnn': (cnn_pred, cnn_conf),
            'prior': prior_score
        }
        
        annotated = self.annotate(frame, res)
        return annotated, res

    def annotate(self, frame: np.ndarray, res: Dict) -> np.ndarray:
        # Implementation based on annotate_pipeline_b from notebook
        out = frame.copy()
        H, W = out.shape[:2]
        label, conf = res['label'], res['confidence']
        bbox, kps = res['bbox'], res['kps']
        
        # Viz settings
        colors_cfg = self.viz_cfg.get('colors', {'ON': (0, 200, 0), 'OFF': (0, 0, 220), 'WARMING': (160, 160, 160)})
        colors = {}
        for k, v in colors_cfg.items():
            if k is True: colors['ON'] = tuple(v)
            elif k is False: colors['OFF'] = tuple(v)
            else: colors[str(k)] = tuple(v)
            
        color = colors.get(label, colors.get('WARMING', (160, 160, 160)))
        overlay_alpha = self.viz_cfg.get('overlay_alpha', 0.6)
        banner_h = self.viz_cfg.get('banner_height', 50)
        font_s = self.viz_cfg.get('font_scale', 0.9)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            
        for kp in kps:
            cv2.circle(out, kp, 5, (255, 200, 0), -1)
            
        # Top banner
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (W, banner_h), color, -1)
        cv2.addWeighted(ov, overlay_alpha, out, 1.0 - overlay_alpha, 0, out)
        cv2.putText(out, f"Seatbelt {label}  ({conf:.0%})", (10, int(banner_h * 0.68)), 
                    cv2.FONT_HERSHEY_DUPLEX, font_s, (255, 255, 255), 2)
                    
        # Score cards (Bottom)
        models = [
            ("YOLO", "ON" if res['yolo'][0] == 1 else "OFF", res['yolo'][1]),
            ("CNN",  "ON" if res['cnn'][0] == 1 else "OFF", res['cnn'][1]),
            ("GEO",  "ON" if res['prior'] >= 0.5 else "OFF", res['prior']),
            ("EMA",  label if label != "WARMING" else "--", conf),
        ]
        
        cell_w, cell_h = 100, 56
        px, py = 8, H - cell_h - 10
        
        for i, (name, pred, score) in enumerate(models):
            cx, cy = px + i * (cell_w + 6), py
            cell_color = colors['ON'] if pred == "ON" else (colors['OFF'] if pred == "OFF" else colors['WARMING'])
            
            ov = out.copy()
            cv2.rectangle(ov, (cx, cy), (cx + cell_w, cy + cell_h), cell_color, -1)
            cv2.addWeighted(ov, overlay_alpha - 0.05, out, 1.05 - overlay_alpha, 0, out)
            cv2.rectangle(out, (cx, cy), (cx + cell_w, cy + cell_h), (210, 210, 210), 1)
            
            cv2.putText(out, name, (cx + 6, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)
            cv2.putText(out, pred, (cx + 6, cy + 36), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 2)
            
            # Progress bar
            bmax = cell_w - 8
            blen = int(bmax * min(max(score, 0.0), 1.0))
            by = cy + cell_h - 7
            cv2.rectangle(out, (cx + 4, by), (cx + 4 + bmax, by + 4), (50, 50, 50), -1)
            cv2.rectangle(out, (cx + 4, by), (cx + 4 + blen, by + 4), (255, 255, 255), -1)
            
        return out
