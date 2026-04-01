import time
import collections
import numpy as np
from src.face_mesh.landmark_utils import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, IRIS_LEFT, IRIS_RIGHT

def get_bounding_box(landmarks: np.ndarray, indices: list):
    """ Get the x,y bounding box for a set of landmarks """
    points = landmarks[indices]
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    return min_x, max_x, min_y, max_y

def compute_gaze_ratios(landmarks: np.ndarray, eye_indices: list, iris_indices: list):
    """ Computes the horizontal and vertical iris position ratios within the eye box. """
    min_x, max_x, min_y, max_y = get_bounding_box(landmarks, eye_indices)
    
    # Iris center is the first point in MediaPipe's iris array subset
    iris_center = landmarks[iris_indices[0]]
    iris_x, iris_y = iris_center[0], iris_center[1]
    
    width = max_x - min_x
    height = max_y - min_y
    
    if width == 0 or height == 0:
        return 0.5, 0.5
        
    horizontal_ratio = (iris_x - min_x) / width
    vertical_ratio = (iris_y - min_y) / height
    
    return horizontal_ratio, vertical_ratio

def determine_gaze_direction(h_ratio: float, v_ratio: float) -> str:
    """ Classify direction based on ratio bounds. """
    if v_ratio > 0.65:
        return "down"
    elif h_ratio < 0.35:
        return "left"
    elif h_ratio > 0.65:
        return "right"
    else:
        return "forward"

class GazeEstimator:
    def __init__(self, thresholds: dict):
        self.alert_seconds = thresholds.get('gaze_alert_seconds', 2.0)
        self.not_forward_start_time = None
        self._ratio_buf = collections.deque(maxlen=5)
        self._stable_direction = "forward"
        self._candidate_direction = None
        self._candidate_count = 0
        self._confirm_frames = 3
        self._last_seen_time = None
        self._hold_after_loss_sec = 0.4

    def _update_stable_direction(self, direction: str) -> str:
        if direction == self._stable_direction:
            self._candidate_direction = None
            self._candidate_count = 0
            return self._stable_direction

        if direction == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = direction
            self._candidate_count = 1

        if self._candidate_count >= self._confirm_frames:
            self._stable_direction = direction
            self._candidate_direction = None
            self._candidate_count = 0

        return self._stable_direction

    def update(self, result_dict: dict) -> dict:
        """
        Calculates gaze direction and tracks off-forward duration.
        """
        if not result_dict.get('valid', False) or result_dict.get('landmarks') is None:
            # Keep last stable gaze briefly during transient invalid frames.
            if self._last_seen_time and (time.time() - self._last_seen_time) <= self._hold_after_loss_sec:
                result_dict['gaze_direction'] = self._stable_direction
            else:
                self.not_forward_start_time = None
                result_dict['gaze_direction'] = None
            return result_dict
            
        landmarks = result_dict['landmarks']
        
        # We average ratios from both eyes to get a stable unified gaze intent
        h_left, v_left = compute_gaze_ratios(landmarks, LEFT_EYE_CONTOUR, IRIS_LEFT)
        h_right, v_right = compute_gaze_ratios(landmarks, RIGHT_EYE_CONTOUR, IRIS_RIGHT)
        
        h_ratio = (h_left + h_right) / 2.0
        v_ratio = (v_left + v_right) / 2.0

        self._ratio_buf.append((h_ratio, v_ratio))
        h_ratio = float(np.mean([p[0] for p in self._ratio_buf]))
        v_ratio = float(np.mean([p[1] for p in self._ratio_buf]))
        
        direction = determine_gaze_direction(h_ratio, v_ratio)
        stable_direction = self._update_stable_direction(direction)
        result_dict['gaze_direction'] = stable_direction
        self._last_seen_time = time.time()
        
        if stable_direction != "forward":
            if self.not_forward_start_time is None:
                self.not_forward_start_time = time.time()
            else:
                duration = time.time() - self.not_forward_start_time
                if duration >= self.alert_seconds:
                    if 'Distraction (Gaze)' not in result_dict['alerts']:
                        result_dict['alerts'].append('Distraction (Gaze)')
        else:
            self.not_forward_start_time = None
            
        return result_dict
