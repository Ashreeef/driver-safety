import time
import numpy as np
from src.face_mesh.landmark_utils import MOUTH_MAR

def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def compute_mar(landmarks: np.ndarray) -> float:
    """ Computes MAR using the same formula structure as EAR on mouth landmarks. """
    p1, p2, p3, p4, p5, p6 = landmarks[MOUTH_MAR]
    
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)
    
    if horizontal == 0:
        return 0.0
    
    mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return mar

class MARTracker:
    def __init__(self, thresholds: dict):
        self.threshold = thresholds.get('mar_threshold', 0.60)
        self.min_seconds = thresholds.get('yawn_min_seconds', 2.0)
        self.yawn_start_time = None

    def update(self, result_dict: dict) -> dict:
        """ Calculates MAR and tracks yawning duration. """
        if not result_dict.get('valid', False) or result_dict.get('landmarks') is None:
            result_dict['mar'] = None
            self.yawn_start_time = None
            return result_dict

        mar = compute_mar(result_dict['landmarks'])
        result_dict['mar'] = mar

        if mar > self.threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = time.time()
            else:
                duration = time.time() - self.yawn_start_time
                if duration >= self.min_seconds:
                    # Alert if sustained
                    if 'Yawning (MAR)' not in result_dict['alerts']:
                        result_dict['alerts'].append('Yawning (MAR)')
        else:
            self.yawn_start_time = None

        return result_dict
