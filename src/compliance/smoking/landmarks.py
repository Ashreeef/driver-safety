import os
import urllib.request
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

_MP_MODEL_URLS = {
    "hand_landmarker.task": 
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task": 
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker_lite.task": 
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

def ensure_model(filename: str, mediapipe_dir: str) -> str:
    """Download the MediaPipe .task file if it does not already exist."""
    path = os.path.join(mediapipe_dir, filename)
    if not os.path.exists(path):
        url = _MP_MODEL_URLS[filename]
        print(f"  ⬇  Downloading {filename} …")
        os.makedirs(mediapipe_dir, exist_ok=True)
        urllib.request.urlretrieve(url, path)
        print(f"  ✅ Saved → {path}")
    return path

class LandmarkExtractor:
    """
    MediaPipe Tasks-only landmark extractor for hand, face, and pose.
    """
    def __init__(self, config: Dict, mediapipe_dir: str = "weights/mediapipe"):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not installed.")

        self.cfg = config
        
        # Ensure models exist
        hand_path = ensure_model("hand_landmarker.task", mediapipe_dir)
        face_path = ensure_model("face_landmarker.task", mediapipe_dir)
        pose_path = ensure_model("pose_landmarker_lite.task", mediapipe_dir)

        # Initialize Hand Landmarker
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_path),
                min_hand_detection_confidence=self.cfg.get('min_hand_detection_confidence', 0.5),
                min_tracking_confidence=self.cfg.get('min_hand_tracking_confidence', 0.5),
                num_hands=2,
            )
        )

        # Initialize Face Landmarker
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=face_path),
                min_face_detection_confidence=self.cfg.get('min_face_detection_confidence', 0.5),
                min_face_presence_confidence=self.cfg.get('min_face_presence_confidence', 0.5),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
        )

        # Initialize Pose Landmarker
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_path),
                min_pose_detection_confidence=self.cfg.get('min_pose_detection_confidence', 0.5),
                min_pose_presence_confidence=self.cfg.get('min_pose_presence_confidence', 0.5),
                min_tracking_confidence=self.cfg.get('min_tracking_confidence', 0.5),
                num_poses=1,
            )
        )

    def extract(self, bgr: np.ndarray) -> Optional[Dict]:
        """Runs hand, face, and pose detection on a frame."""
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = self.hand_detector.detect(mp_image)
        face_result = self.face_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)

        # Face is required for the smoking signal (mouth center)
        if not face_result.face_landmarks:
            return None

        out: Dict[str, np.ndarray] = {"frame_wh": (W, H)}

        # Face: mouth centre (landmarks 13 & 14)
        face = face_result.face_landmarks[0]
        out["mouth_centre"] = (
            np.array([face[13].x, face[13].y]) +
            np.array([face[14].x, face[14].y])
        ) / 2.0

        # Hand: wrist + fingertips
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            out["right_wrist"]     = np.array([hand[0].x, hand[0].y])
            out["right_index_tip"] = np.array([hand[8].x, hand[8].y])
            out["right_thumb_tip"] = np.array([hand[4].x, hand[4].y])

        # Pose: right shoulder (12) and right elbow (14)
        if pose_result.pose_landmarks:
            pose = pose_result.pose_landmarks[0]
            out["right_shoulder"] = np.array([pose[12].x, pose[12].y])
            out["right_elbow"]    = np.array([pose[14].x, pose[14].y])
            if "right_wrist" not in out:
                # Fallback to pose wrist if hand detector missed it
                out["right_wrist"] = np.array([pose[16].x, pose[16].y])

        return out

    @staticmethod
    def hand_mouth_distance(lm_dict: Dict) -> float:
        """Normalised wrist-to-mouth distance."""
        if "right_wrist" not in lm_dict or "mouth_centre" not in lm_dict:
            return 1.0
        return float(np.linalg.norm(lm_dict["right_wrist"] - lm_dict["mouth_centre"]))

    @staticmethod
    def elbow_angle(lm_dict: Dict) -> Optional[float]:
        """Elbow flexion angle in degrees."""
        keys = ("right_shoulder", "right_elbow", "right_wrist")
        if not all(k in lm_dict for k in keys):
            return None
        a = lm_dict["right_shoulder"] - lm_dict["right_elbow"]
        b = lm_dict["right_wrist"]    - lm_dict["right_elbow"]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def proximity_score(self, lm_dict: Dict) -> float:
        """Maps distance to [0, 1] score based on threshold."""
        dist_thresh = self.cfg.get('hand_mouth_dist_thresh', 0.15)
        d = self.hand_mouth_distance(lm_dict)
        return float(np.clip(1.0 - d / dist_thresh, 0.0, 1.0))
