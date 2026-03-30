import numpy as np
from src.face_mesh.landmark_utils import RIGHT_EYE_EAR, LEFT_EYE_EAR

def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_ear(landmarks: np.ndarray, eye_indices: list) -> float:
    """ Computes EAR using Euclidean distance ratio. """
    p1, p2, p3, p4, p5, p6 = landmarks[eye_indices]
    
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    
    if horizontal == 0:
        return 0.0
        
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

class EARTracker:
    def __init__(self, thresholds: dict):
        self.threshold = thresholds.get('ear_threshold', 0.20)
        self.consec_frames = thresholds.get('ear_consec_frames', 4)
        self.counter = 0

    def update(self, result_dict: dict) -> dict:
        """
        Calculates EAR and updates the state. Populates the result_dict.
        """
        if not result_dict.get('valid', False) or result_dict.get('landmarks') is None:
            result_dict['ear'] = None
            self.counter = 0
            return result_dict

        landmarks = result_dict['landmarks']
        left_ear = compute_ear(landmarks, LEFT_EYE_EAR)
        right_ear = compute_ear(landmarks, RIGHT_EYE_EAR)
        avg_ear = (left_ear + right_ear) / 2.0

        result_dict['ear'] = avg_ear

        if avg_ear < self.threshold:
            self.counter += 1
            if self.counter >= self.consec_frames:
                result_dict['alerts'].append('Drowsiness (EAR)')
        else:
            self.counter = 0

        return result_dict
