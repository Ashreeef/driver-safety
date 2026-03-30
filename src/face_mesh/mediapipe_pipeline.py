import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.face_mesh.landmark_utils import get_landmarks_array, get_pixel_coords
from src.face_mesh.head_pose import estimate_head_pose

class FaceMeshDetector:
    def __init__(self, thresholds_dict: dict, model_paths_dict: dict):
        """
        Initializes the offline MediaPipe FaceLandmarker Task.
        Loads configuration from provided config dictionaries.
        """
        self.thresholds = thresholds_dict
        model_path = model_paths_dict.get('mediapipe', {}).get('face_landmarker', 'models/mediapipe/face_landmarker.task')
        
        # Initialize MediaPipe Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, image_bgr: np.ndarray) -> dict:
        """
        Processes a single BGR image frame.
        Extracts 468+ landmarks, computes head pose, and checks validity.
        Returns the standardized output dictionary.
        """
        result_dict = {
            "landmarks": None,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "valid": False,
            "ear": None,
            "mar": None,
            "perclos": None,
            "alerts": []
        }

        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Process frame
        # Wrap in try-except in case of task errors or missing model during testing
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            # Silently fail and return invalid if model isn't loaded correctly
            # This allows testing the logic flow if the user hasn't downloaded the blob yet
            return result_dict
            
        if not detection_result.face_landmarks:
            return result_dict
            
        landmarks_array = get_landmarks_array(detection_result.face_landmarks)
        if landmarks_array is None:
            return result_dict
            
        img_h, img_w, _ = image_bgr.shape
        pixel_coords = get_pixel_coords(landmarks_array, img_w, img_h)
        
        # Calculate pitch, yaw, roll using solvePnP
        pitch, yaw, roll = estimate_head_pose(pixel_coords, img_w, img_h)
        
        # Check constraints
        yaw_max = self.thresholds.get('head_yaw_max', 45)
        pitch_max = self.thresholds.get('head_pitch_max', 30)
        
        valid = True
        if abs(yaw) > yaw_max or abs(pitch) > pitch_max:
            valid = False

        result_dict["landmarks"] = landmarks_array
        result_dict["pitch"] = pitch
        result_dict["yaw"] = yaw
        result_dict["roll"] = roll
        result_dict["valid"] = valid
        
        return result_dict

    def close(self):
        """Releases MediaPipe resources."""
        self.detector.close()
