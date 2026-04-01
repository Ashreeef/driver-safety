import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.face_mesh.landmark_utils import (
    get_landmarks_array,
    get_pixel_coords,
    LandmarkStabilizer,
)
from src.face_mesh.head_pose import estimate_head_pose


class FaceMeshDetector:
    """
    Module 1: Face Mesh + Head Pose.

    Pipeline per frame:
      1. MediaPipe detects face landmarks (478 pts including iris).
      2. LandmarkStabilizer applies EMA to reduce jitter.
      3. Head pose estimated via solvePnP with transform_matrix fallback.
      4. Camera mount offset subtracted from yaw/pitch.
      5. validity_flag computed from corrected angles.

    The returned result_dict is the contract shared by all downstream modules.
    """

    def __init__(self, thresholds_dict: dict, model_paths_dict: dict):
        self.thresholds = thresholds_dict

        # Camera mount offset (degrees) — calibrate to your physical setup
        self._yaw_offset   = thresholds_dict.get('camera_yaw_offset',   0.0)
        self._pitch_offset = thresholds_dict.get('camera_pitch_offset',  0.0)
        self._yaw_max      = thresholds_dict.get('head_yaw_max',  60.0)
        self._pitch_max    = thresholds_dict.get('head_pitch_max', 40.0)

        # Landmark stabilizer
        alpha = thresholds_dict.get('landmark_ema_alpha', 0.4)
        self._stabilizer = LandmarkStabilizer(alpha=alpha)

        # Frames since last successful face detection (for stabilizer reset)
        self._no_face_frames = 0
        self._reset_after    = int(thresholds_dict.get('camera_fps', 15))  # 1 sec

        # MediaPipe Tasks API
        model_path = (model_paths_dict
                      .get('mediapipe', {})
                      .get('face_landmarker',
                           'models/mediapipe/face_landmarker.task'))

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,  # needed for fallback
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------

    def process_frame(self, image_bgr: np.ndarray) -> dict:
        result_dict = {
            "landmarks":        None,
            "pitch":            0.0,
            "yaw":              0.0,
            "roll":             0.0,
            "valid":            False,
            "ear":              None,
            "ear_raw":          None,
            "ear_calibrated":   False,
            "ear_baseline":     None,
            "mar":              None,
            "perclos":          None,
            "gaze_direction":   None,
            "yawn_count":       0,
            "head_pose_method": "none",
            "alerts":           [],
        }

        # --- MediaPipe detection ---
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        try:
            detection = self.detector.detect(mp_image)
        except Exception:
            return result_dict

        if not detection.face_landmarks:
            self._no_face_frames += 1
            if self._no_face_frames >= self._reset_after:
                self._stabilizer.reset()   # clear EMA state after 1s absence
            return result_dict

        self._no_face_frames = 0

        # --- Landmark extraction + stabilization ---
        raw_lm = get_landmarks_array(detection.face_landmarks)
        if raw_lm is None:
            return result_dict

        landmarks = self._stabilizer.smooth(raw_lm)   # EMA applied here

        img_h, img_w = image_bgr.shape[:2]
        pixel_coords = get_pixel_coords(landmarks, img_w, img_h)

        # --- Head pose (solvePnP → transform_matrix fallback) ---
        transform_matrix = (
            detection.facial_transformation_matrixes[0]
            if detection.facial_transformation_matrixes
            else None
        )

        pitch, yaw, roll, method = estimate_head_pose(
            pixel_coords, img_w, img_h, transform_matrix
        )

        # Subtract camera mount offset
        yaw   -= self._yaw_offset
        pitch -= self._pitch_offset

        # --- Validity check ---
        valid = (abs(yaw) <= self._yaw_max and abs(pitch) <= self._pitch_max)

        result_dict.update({
            "landmarks":        landmarks,
            "pitch":            pitch,
            "yaw":              yaw,
            "roll":             roll,
            "valid":            valid,
            "head_pose_method": method,
        })
        return result_dict

    def close(self):
        self.detector.close()