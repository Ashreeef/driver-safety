import os
import cv2
import numpy as np
from typing import Dict, Optional

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class LandmarkExtractor:
    """
    MediaPipe landmark extractor for the smoking detection pipeline.

    Two operating modes, selected by use_external_face_landmarks:

      True  (integrated mode, default):
        The caller supplies result_dict['landmarks'] as the face_landmarks
        argument to extract(). The face landmarker is NOT initialised here,
        saving one full MediaPipe inference per frame.  Only HandLandmarker
        and PoseLandmarker are created (2 inferences / frame).

      False (standalone mode):
        No external face landmarks are available.  A FaceLandmarker is
        initialised internally and run each frame (3 inferences / frame).
        Use this only for standalone smoking_inference.py testing.

    Model paths must be provided explicitly (from model_paths.yaml).
    Models are never downloaded at runtime — place the .task files under
    models/mediapipe/ before first use.
    """

    def __init__(self, config: Dict, model_paths: Dict,
                 use_external_face_landmarks: bool = True):
        """
        Args:
            config:                      landmarks section of smoking.yaml
            model_paths:                 mediapipe section of model_paths.yaml
            use_external_face_landmarks: True  → integrated mode (no face detector)
                                         False → standalone mode (face detector created)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not installed.")

        self.cfg = config
        self._external_face = use_external_face_landmarks

        hand_path = model_paths.get('hand_landmarker', '')
        pose_path = model_paths.get('pose_landmarker', '')

        if not os.path.exists(hand_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at '{hand_path}'. "
                "Place the .task file there before running."
            )
        if not os.path.exists(pose_path):
            raise FileNotFoundError(
                f"Pose landmarker model not found at '{pose_path}'. "
                "Place the .task file there before running."
            )

        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_path),
                min_hand_detection_confidence=self.cfg.get('min_hand_detection_confidence', 0.5),
                min_tracking_confidence=self.cfg.get('min_hand_tracking_confidence', 0.5),
                num_hands=2,
            )
        )

        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_path),
                min_pose_detection_confidence=self.cfg.get('min_pose_detection_confidence', 0.5),
                min_pose_presence_confidence=self.cfg.get('min_pose_presence_confidence', 0.5),
                min_tracking_confidence=self.cfg.get('min_tracking_confidence', 0.5),
                num_poses=1,
            )
        )

        # Face detector: only in standalone mode
        self.face_detector = None
        if not use_external_face_landmarks:
            face_path = model_paths.get('face_landmarker', '')
            if not os.path.exists(face_path):
                raise FileNotFoundError(
                    f"Face landmarker model not found at '{face_path}'. "
                    "Place the .task file there before running."
                )
            self.face_detector = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=face_path),
                    min_face_detection_confidence=self.cfg.get('min_face_detection_confidence', 0.5),
                    min_face_presence_confidence=self.cfg.get('min_face_presence_confidence', 0.5),
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
            )

    def extract(self, bgr: np.ndarray,
                face_landmarks: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Run hand and pose detection; derive face data from provided or
        internally-detected landmarks.

        Args:
            bgr:            Current frame in BGR format.
            face_landmarks: result_dict['landmarks'] — shape (478, 3), normalised
                            [0, 1].  Required in integrated mode.  Ignored in
                            standalone mode (face detector runs internally).

        Returns:
            dict with keys: frame_wh, mouth_centre, right_wrist, right_index_tip,
            right_thumb_tip, right_shoulder, right_elbow.
            Returns None if no face is available (no landmarks passed and face
            not detected internally).
        """
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = self.hand_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)

        out: Dict = {"frame_wh": (W, H)}

        # Resolve mouth centre ------------------------------------------------
        if self._external_face:
            # Integrated mode: face_landmarks is result_dict['landmarks'] (478, 3)
            if face_landmarks is None:
                return None
            out["mouth_centre"] = (face_landmarks[13, :2] + face_landmarks[14, :2]) / 2.0
        else:
            # Standalone mode: run internal face detector
            face_result = self.face_detector.detect(mp_image)
            if not face_result.face_landmarks:
                return None
            face = face_result.face_landmarks[0]
            out["mouth_centre"] = (
                np.array([face[13].x, face[13].y]) +
                np.array([face[14].x, face[14].y])
            ) / 2.0

        # Hand: wrist + fingertips --------------------------------------------
        if hand_result.hand_landmarks:
            hand = hand_result.hand_landmarks[0]
            out["right_wrist"]     = np.array([hand[0].x, hand[0].y])
            out["right_index_tip"] = np.array([hand[8].x, hand[8].y])
            out["right_thumb_tip"] = np.array([hand[4].x, hand[4].y])

        # Pose: shoulder + elbow for elbow-angle bonus ------------------------
        if pose_result.pose_landmarks:
            pose = pose_result.pose_landmarks[0]
            out["right_shoulder"] = np.array([pose[12].x, pose[12].y])
            out["right_elbow"]    = np.array([pose[14].x, pose[14].y])
            if "right_wrist" not in out:
                out["right_wrist"] = np.array([pose[16].x, pose[16].y])

        return out

    @staticmethod
    def hand_mouth_distance(lm_dict: Dict) -> float:
        if "right_wrist" not in lm_dict or "mouth_centre" not in lm_dict:
            return 1.0
        return float(np.linalg.norm(lm_dict["right_wrist"] - lm_dict["mouth_centre"]))

    @staticmethod
    def elbow_angle(lm_dict: Dict) -> Optional[float]:
        keys = ("right_shoulder", "right_elbow", "right_wrist")
        if not all(k in lm_dict for k in keys):
            return None
        a = lm_dict["right_shoulder"] - lm_dict["right_elbow"]
        b = lm_dict["right_wrist"]    - lm_dict["right_elbow"]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def proximity_score(self, lm_dict: Dict) -> float:
        dist_thresh = self.cfg.get('hand_mouth_dist_thresh', 0.15)
        d = self.hand_mouth_distance(lm_dict)
        return float(np.clip(1.0 - d / dist_thresh, 0.0, 1.0))
