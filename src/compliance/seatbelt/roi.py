import cv2
import numpy as np
import os
from typing import Tuple, List, Optional, Union, Dict

# MediaPipe imports
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python.vision import pose_landmarker, RunningMode
    from mediapipe import Image, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class ROIExtractor:
    """
    Standard interface for extracting regions of interest (ROI) from frames.
    Supports MediaPipe Pose landmarks and YOLO-based detection.

    model_path must be supplied explicitly (from model_paths.yaml).
    Models are never downloaded at runtime — place the .task file under
    models/mediapipe/ before first use.
    """

    def __init__(self, method: str = 'mediapipe', model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.method = method
        self.model_path = model_path
        self.config = config or {}
        self.pose_estimator = None

        if method == 'mediapipe' and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe(model_path)

    def _init_mediapipe(self, model_path: Optional[str]):
        if model_path is None:
            raise ValueError(
                "ROIExtractor requires an explicit model_path. "
                "Pass model_paths['mediapipe']['pose_landmarker'] from model_paths.yaml."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at '{model_path}'. "
                "Place the .task file there before running."
            )

        min_presence = self.config.get('min_pose_presence_confidence', 0.5)
        min_detection = self.config.get('min_pose_detection_confidence', 0.5)
        min_tracking = self.config.get('min_tracking_confidence', 0.5)
        
        options = pose_landmarker.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_pose_presence_confidence=min_presence,
            min_pose_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking
        )
        self.pose_estimator = pose_landmarker.PoseLandmarker.create_from_options(options)

    def extract(self, frame: np.ndarray, padding: float = 0.2) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], List]:
        """
        Main entry point for ROI extraction.
        
        Returns:
            roi: The cropped image (BGR) or None
            bbox: (x1, y1, x2, y2) pixel coordinates
            keypoints: Method-specific keypoints (e.g., MediaPipe landmarks)
        """
        if self.method == 'mediapipe' and self.pose_estimator:
            return self._extract_mediapipe(frame, padding)
        else:
            # Fallback or other methods (YOLOv5 ROI is usually handled during inference loop)
            H, W = frame.shape[:2]
            return frame, (0, 0, W, H), []

    def _extract_mediapipe(self, frame: np.ndarray, padding: float) -> Tuple[np.ndarray, Tuple[int, int, int, int], List]:
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        
        results = self.pose_estimator.detect(mp_image)
        
        if not results or not results.pose_landmarks:
            return frame, (0, 0, W, H), []

        lm_list = results.pose_landmarks[0]
        # Externalized indices for torso (default: shoulders (11, 12) and hips (23, 24))
        torso_indices = self.config.get('torso_indices', [11, 12, 23, 24])
        pts = [(lm_list[i].x, lm_list[i].y) for i in torso_indices]
        kp_pixels = [(int(x * W), int(y * H)) for x, y in pts]
        
        xs = [p[0] for p in kp_pixels]
        ys = [p[1] for p in kp_pixels]
        x1_raw, y1_raw = min(xs), min(ys)
        x2_raw, y2_raw = max(xs), max(ys)
        
        pad_x = int((x2_raw - x1_raw) * padding)
        pad_y = int((y2_raw - y1_raw) * padding)
        
        x1 = max(0, x1_raw - pad_x)
        y1 = max(0, y1_raw - pad_y)
        x2 = min(W, x2_raw + pad_x)
        y2 = min(H, y2_raw + pad_y)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame, (0, 0, W, H), kp_pixels
            
        return roi, (x1, y1, x2, y2), kp_pixels

def get_yolo_roi_boxes(yolo_model, frame: np.ndarray, conf_thresh: float = 0.45) -> List[Tuple[int, int, int, int, float]]:
    """Helper for YOLOv5/v8 ROI extraction (Pipeline A style)."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Check if YOLOv5 (torch.hub) or YOLOv8 (ultralytics)
    model_type = type(yolo_model).__name__
    is_yolov5 = "AutoShape" in model_type or hasattr(yolo_model, 'model') and "common" in str(type(yolo_model))
    
    if is_yolov5: # YOLOv5 (torch.hub)
        # YOLOv5 AutoShape doesn't take 'verbose' or 'conf' in forward call.
        # It uses internal attributes or post-filtering.
        res = yolo_model(img_rgb)
        if hasattr(res, 'xyxy'):
            results = res.xyxy[0].cpu().numpy()
        else:
            # Fallback if structure is different
            return []
            
        boxes = []
        for det in results:
            x1, y1, x2, y2, conf, cls = det
            if conf >= conf_thresh:
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return boxes
    else: # YOLOv8/v10/etc via ultralytics class
        try:
            results = yolo_model(frame, verbose=False, conf=conf_thresh)
        except TypeError:
            # Handle cases where 'verbose' is not supported in the underlying model's fuse method
            results = yolo_model(frame, conf=conf_thresh)
            
        boxes = []
        for r in results:
            if r.boxes:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    conf = b.conf[0].item()
                    boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return boxes
