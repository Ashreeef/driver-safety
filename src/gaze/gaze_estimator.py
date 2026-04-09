import time
import collections
import numpy as np
from src.face_mesh.landmark_utils import (
    LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, IRIS_LEFT, IRIS_RIGHT
)


def get_bounding_box(landmarks: np.ndarray, indices: list):
    points = landmarks[indices]
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    return min_x, max_x, min_y, max_y


def compute_gaze_ratios(landmarks: np.ndarray,
                        eye_indices: list,
                        iris_indices: list):
    min_x, max_x, min_y, max_y = get_bounding_box(landmarks, eye_indices)
    iris_center = landmarks[iris_indices[0]]
    iris_x, iris_y = iris_center[0], iris_center[1]

    width  = max_x - min_x
    height = max_y - min_y

    if width == 0 or height == 0:
        return 0.5, 0.5

    h_ratio = (iris_x - min_x) / width
    v_ratio = (iris_y - min_y) / height
    return h_ratio, v_ratio


def determine_gaze_direction(h_ratio: float, v_ratio: float) -> str:
    """
    Classify gaze from corrected iris ratios.

    Zones are tightened vs original [0.35, 0.65] so that eye-only movement
    (smaller signal than a full head turn) can cross the boundary.
    'up' is added — was missing entirely before.
    """
    if v_ratio < 0.35:          # looking up
        return "up"
    if v_ratio > 0.65:          # looking down
        return "down"
    if h_ratio < 0.40:
        return "left"
    if h_ratio > 0.60:
        return "right"
    return "forward"


class GazeEstimator:
    def __init__(self, thresholds: dict):
        self.alert_seconds        = thresholds.get('gaze_alert_seconds', 2.0)
        self._hold_after_loss_sec = thresholds.get('gaze_hold_after_loss_sec', 0.4)
        self._confirm_frames      = thresholds.get('gaze_confirm_frames', 3)

        # Head-pose correction scales.
        # Yaw > 0 (head right) shifts iris ratio leftward → we add back yaw * scale.
        # Pitch > 0 (head down) shifts iris ratio downward → we subtract pitch * scale.
        # Signs must match your solvePnP convention — flip sign in YAML if needed.
        self._yaw_scale   = thresholds.get('gaze_yaw_correction_scale',   0.008)
        self._pitch_scale = thresholds.get('gaze_pitch_correction_scale', 0.005)

        self.not_forward_start_time = None
        self._ratio_buf           = collections.deque(maxlen=5)
        self._stable_direction    = "forward"
        self._candidate_direction = None
        self._candidate_count     = 0
        self._last_seen_time      = None

    def _update_stable_direction(self, direction: str) -> str:
        if direction == self._stable_direction:
            self._candidate_direction = None
            self._candidate_count     = 0
            return self._stable_direction

        if direction == self._candidate_direction:
            self._candidate_count += 1
        else:
            self._candidate_direction = direction
            self._candidate_count     = 1

        if self._candidate_count >= self._confirm_frames:
            self._stable_direction    = direction
            self._candidate_direction = None
            self._candidate_count     = 0

        return self._stable_direction

    def update(self, result_dict: dict) -> dict:
        if not result_dict.get('valid', False) or \
                result_dict.get('landmarks') is None:
            if self._last_seen_time and \
                    (time.time() - self._last_seen_time) <= self._hold_after_loss_sec:
                result_dict['gaze_direction'] = self._stable_direction
            else:
                self.not_forward_start_time = None
                result_dict['gaze_direction'] = None
            return result_dict

        landmarks = result_dict['landmarks']

        # --- raw iris ratios for both eyes ---
        h_left,  v_left  = compute_gaze_ratios(
            landmarks, LEFT_EYE_CONTOUR,  IRIS_LEFT)
        h_right, v_right = compute_gaze_ratios(
            landmarks, RIGHT_EYE_CONTOUR, IRIS_RIGHT)

        h_ratio = (h_left + h_right) / 2.0
        v_ratio = (v_left + v_right) / 2.0

        # --- 5-frame smoothing ---
        self._ratio_buf.append((h_ratio, v_ratio))
        h_ratio = float(np.mean([p[0] for p in self._ratio_buf]))
        v_ratio = float(np.mean([p[1] for p in self._ratio_buf]))

        # --- head-pose correction ---
        # Subtract the portion of the iris shift caused by head rotation,
        # leaving only the eyeball-rotation component.
        yaw   = result_dict.get('yaw',   0.0)
        pitch = result_dict.get('pitch', 0.0)
        h_ratio = h_ratio + yaw   * self._yaw_scale
        v_ratio = v_ratio - pitch * self._pitch_scale

        # --- classify & stabilize ---
        direction        = determine_gaze_direction(h_ratio, v_ratio)
        stable_direction = self._update_stable_direction(direction)
        result_dict['gaze_direction'] = stable_direction
        self._last_seen_time = time.time()

        # --- alert timer ---
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