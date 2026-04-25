import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
from ultralytics import YOLO

from .landmarks import LandmarkExtractor

@dataclass
class SmokingFrameResult:
    """Full pipeline output for a single frame."""
    landmark_score:  float
    detection_score: float
    fusion_score:    float
    temporal_conf:   float
    smoking_flag:    bool
    alert_active:    bool
    detections:      List[Tuple[str, float, Tuple[int, int, int, int]]]
    landmarks:       Optional[Dict]

class SmokingDetector:
    """
    Hybrid smoking detector:
    - Branch A: Landmark-based (MediaPipe)
    - Branch B: Detection-based (YOLO)
    - Fusion: Weighted score-level
    - Temporal: Sliding-window majority vote
    """
    def __init__(self, config: Dict, extractor: Optional[LandmarkExtractor], yolo_weights: Optional[str] = None):
        self.cfg = config
        self.extractor = extractor
        self.yolo = None

        if yolo_weights and os.path.exists(yolo_weights):
            try:
                self.yolo = YOLO(yolo_weights)
                print(f"  ✅ Smoking YOLO loaded from {yolo_weights}")
            except Exception as e:
                print(f"  ⚠  Failed to load smoking YOLO: {e}")
        
        # Temporal buffers
        window_size = self.cfg.get('temporal_buffer', {}).get('window_size', 8)
        self.buffer = deque(maxlen=window_size)
        self.alert_active = False
        self._prev_wrist = None
        self.frame_count = 0

    def reset(self):
        self.buffer.clear()
        self.alert_active = False
        self._prev_wrist = None
        self.frame_count = 0

    def _landmark_score(self, bgr: np.ndarray) -> Tuple[float, Optional[Dict]]:
        if not self.extractor:
            return 0.0, None
        
        lm = self.extractor.extract(bgr)
        if lm is None:
            return 0.0, None
        
        prox_score = self.extractor.proximity_score(lm)
        
        # Elbow bonus
        bonus = 0.0
        angle = LandmarkExtractor.elbow_angle(lm)
        lm_cfg = self.cfg.get('landmarks', {})
        if angle is not None:
            if lm_cfg.get('elbow_angle_min', 30.0) < angle < lm_cfg.get('elbow_angle_max', 150.0):
                bonus = 0.15
        
        # Velocity penalty
        penalty = 0.0
        curr_wrist = lm.get("right_wrist")
        if curr_wrist is not None and self._prev_wrist is not None:
            dist = float(np.linalg.norm(curr_wrist - self._prev_wrist))
            if dist > lm_cfg.get('wrist_velocity_thresh', 0.25):
                penalty = 0.20
        self._prev_wrist = curr_wrist.copy() if curr_wrist is not None else None
        
        score = float(np.clip(prox_score + bonus - penalty, 0.0, 1.0))
        return score, lm

    def _detection_score(self, bgr: np.ndarray) -> Tuple[float, List]:
        if not self.yolo:
            return 0.0, []
        
        det_cfg = self.cfg.get('detection', {})
        try:
            results = self.yolo(
                bgr, 
                conf=det_cfg.get('conf_thresh', 0.45),
                iou=det_cfg.get('iou_thresh', 0.45),
                verbose=False
            )[0]
        except Exception:
            return 0.0, []

        detections = []
        best_conf = 0.0
        class_map = det_cfg.get('classes', {})
        
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = class_map.get(cls_id, "unknown")
            conf = float(box.conf)
            bbox = tuple(box.xyxy[0].cpu().numpy().astype(int))
            detections.append((cls_name, conf, bbox))
            best_conf = max(best_conf, conf)
            
        return best_conf, detections

    def process(self, bgr: np.ndarray) -> SmokingFrameResult:
        s_L, lm = self._landmark_score(bgr)
        s_D, dets = self._detection_score(bgr)
        
        # Fusion
        alpha = self.cfg.get('fusion', {}).get('alpha', 0.10)
        s_F = alpha * s_L + (1.0 - alpha) * s_D
        
        # Temporal
        self.buffer.append(s_F >= 0.5)
        temp_cfg = self.cfg.get('temporal_buffer', {})
        pos_count = sum(self.buffer)
        temporal_conf = pos_count / max(len(self.buffer), 1)
        
        confirm_thresh = temp_cfg.get('confirm_threshold', 5)
        hysteresis_low = temp_cfg.get('hysteresis_low', 0.30)
        
        if pos_count >= confirm_thresh:
            self.alert_active = True
        elif temporal_conf < hysteresis_low:
            self.alert_active = False
            
        res = SmokingFrameResult(
            landmark_score=s_L,
            detection_score=s_D,
            fusion_score=s_F,
            temporal_conf=temporal_conf,
            smoking_flag=(pos_count >= confirm_thresh),
            alert_active=self.alert_active,
            detections=dets,
            landmarks=lm
        )
        self.frame_count += 1
        return res
