import time
import collections
import numpy as np
from src.face_mesh.landmark_utils import (
    MOUTH_MAR_TOP, MOUTH_MAR_BOTTOM, MOUTH_MAR_LEFT, MOUTH_MAR_RIGHT
)


def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_mar(landmarks: np.ndarray) -> float:
    """
    Computes MAR as vertical lip distance / horizontal mouth width.

    Uses landmark 13 (upper lip center) and 14 (lower lip center) because
    these are the extreme vertical points of the lips — maximum signal for
    yawn detection. Previous indices (82, 87, 312, 317) were near the
    corners and measured roughly half the actual jaw opening.

    Typical values:
      closed mouth : 0.05 – 0.15
      speaking     : 0.20 – 0.35
      mild yawn    : 0.35 – 0.50
      wide yawn    : 0.50 – 0.75
    """
    vertical   = euclidean_dist(
        landmarks[MOUTH_MAR_TOP], landmarks[MOUTH_MAR_BOTTOM]
    )
    horizontal = euclidean_dist(
        landmarks[MOUTH_MAR_LEFT], landmarks[MOUTH_MAR_RIGHT]
    )
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


class MARTracker:
    """
    Tracks MAR, detects yawns with a duration filter, and accumulates
    yawn frequency over a rolling time window.

    Alert logic:
      - 'Yawning (MAR)'         : single yawn confirmed (MAR > threshold for >= min_seconds)
      - 'Fatigue (Yawn Frequency)': >= yawn_count_alert yawns within yawn_window_seconds
    """

    def __init__(self, thresholds: dict):
        self.threshold        = thresholds.get('mar_threshold', 0.45)
        self.min_seconds      = thresholds.get('yawn_min_seconds', 2.5)
        self.window_seconds   = thresholds.get('yawn_window_seconds', 300)   # 5 min
        self.count_alert      = thresholds.get('yawn_count_alert', 3)

        # Duration tracking — detects a single yawn
        self.yawn_start_time   = None
        self._yawn_in_progress = False   # True while MAR is above threshold
        self._yawn_alerted     = False   # prevents repeat alert for same yawn event

        # Frequency accumulator — rolling window of confirmed yawn timestamps
        self.yawn_timestamps: collections.deque = collections.deque()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_yawn(self):
        """Called once per confirmed yawn event. Updates the rolling window."""
        now = time.time()
        self.yawn_timestamps.append(now)
        # Evict yawns older than the window
        while self.yawn_timestamps and \
              (now - self.yawn_timestamps[0]) > self.window_seconds:
            self.yawn_timestamps.popleft()

    def _yawn_count_in_window(self) -> int:
        now = time.time()
        while self.yawn_timestamps and \
              (now - self.yawn_timestamps[0]) > self.window_seconds:
            self.yawn_timestamps.popleft()
        return len(self.yawn_timestamps)

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, result_dict: dict) -> dict:
        """
        Call after FaceMeshDetector.process_frame().
        Mutates result_dict in-place and returns it.
        """
        if not result_dict.get('valid', False) or \
           result_dict.get('landmarks') is None:
            # Reset on invalid frame — avoids ghost yawn after head turn
            result_dict['mar'] = None
            self._reset_duration()
            return result_dict

        mar = compute_mar(result_dict['landmarks'])
        result_dict['mar'] = mar

        if mar > self.threshold:
            # --- Mouth is open ---
            if not self._yawn_in_progress:
                # Leading edge of a potential yawn
                self._yawn_in_progress = True
                self.yawn_start_time   = time.time()
                self._yawn_alerted     = False
            else:
                # Mouth has been open since yawn_start_time
                duration = time.time() - self.yawn_start_time
                if duration >= self.min_seconds and not self._yawn_alerted:
                    # Confirmed yawn
                    result_dict['alerts'].append('Yawning (MAR)')
                    self._record_yawn()
                    self._yawn_alerted = True   # don't re-alert same event

                    # Check frequency threshold
                    if self._yawn_count_in_window() >= self.count_alert:
                        result_dict['alerts'].append('Fatigue (Yawn Frequency)')
        else:
            # --- Mouth closed ---
            self._reset_duration()

        # Always expose current yawn count for the overlay
        yawn_count = self._yawn_count_in_window()
        result_dict['yawn_count'] = yawn_count
        return result_dict

    def _reset_duration(self):
        self._yawn_in_progress = False
        self.yawn_start_time   = None
        self._yawn_alerted     = False