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


class GazeEstimator:
    """
    Gaze estimation using iris position relative to eye bounding box.

    Key design decisions:
    - Uses DEVIATION from a calibrated neutral reference, not absolute thresholds.
      This handles per-person iris anatomy differences (some people's iris
      naturally sits high or low in the bounding box).
    - Head-pose correction is kept small and optional. The old scales (0.008/0.005)
      were large enough to completely override the iris signal. They are now set
      to 0.0 by default until validated.
    - Vertical axis: MediaPipe y=0 is top of image, y=1 is bottom.
      Iris moves UP in frame (smaller y) → looking up → negative v_deviation.
      Iris moves DOWN in frame (larger y) → looking down → positive v_deviation.
    """

    # How many frames to collect for neutral calibration
    CALIB_FRAMES = 40

    def __init__(self, thresholds: dict):
        self.alert_seconds        = thresholds.get('gaze_alert_seconds', 2.0)
        self._hold_after_loss_sec = thresholds.get('gaze_hold_after_loss_sec', 0.4)
        self._confirm_frames      = thresholds.get('gaze_confirm_frames', 5)

        # Deviation thresholds — how far from neutral before direction changes.
        # Horizontal: symmetric. Vertical: looking down is easier than looking up
        # (eyelid covers upward gaze more), so make them independent.
        self._h_thresh   = thresholds.get('gaze_h_threshold', 0.12)
        self._v_up_thresh   = thresholds.get('gaze_v_up_threshold', 0.10)
        self._v_down_thresh = thresholds.get('gaze_v_down_threshold', 0.13)

        # Head-pose correction — set to 0 until you verify iris tracking works
        # on its own. Only enable after confirming iris-only detection is correct.
        # To enable: set gaze_yaw_correction_scale to 0.003 in thresholds.yaml.
        self._yaw_scale   = thresholds.get('gaze_yaw_correction_scale',   0.0)
        self._pitch_scale = thresholds.get('gaze_pitch_correction_scale', 0.0)

        # Neutral reference — set during calibration
        self._neutral_h  = None   # h_ratio when looking forward
        self._neutral_v  = None   # v_ratio when looking forward
        self._calib_h_buf = []
        self._calib_v_buf = []
        self._calibrated  = False

        # Smoothing buffer
        self._ratio_buf = collections.deque(maxlen=5)

        # Direction stabilizer
        self._stable_direction    = "forward"
        self._candidate_direction = None
        self._candidate_count     = 0

        # Alert timer
        self.not_forward_start_time = None
        self._last_seen_time        = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _feed_calibration(self, h: float, v: float):
        """
        Collect neutral-gaze samples. Called during the first CALIB_FRAMES
        valid frames. Driver should be looking at the road/camera during this.
        """
        self._calib_h_buf.append(h)
        self._calib_v_buf.append(v)
        if len(self._calib_h_buf) >= self.CALIB_FRAMES:
            self._neutral_h  = float(np.median(self._calib_h_buf))
            self._neutral_v  = float(np.median(self._calib_v_buf))
            self._calibrated = True
            print(
                f"[GazeEstimator] Calibrated. "
                f"neutral_h={self._neutral_h:.3f}  "
                f"neutral_v={self._neutral_v:.3f}"
            )

    def calibration_progress(self) -> float:
        return min(len(self._calib_h_buf) / self.CALIB_FRAMES, 1.0)

    def reset_calibration(self):
        """Call this to force re-calibration (e.g. driver change)."""
        self._calib_h_buf.clear()
        self._calib_v_buf.clear()
        self._calibrated  = False
        self._neutral_h   = None
        self._neutral_v   = None
        self._ratio_buf.clear()

    # ------------------------------------------------------------------
    # Direction classification (deviation-based)
    # ------------------------------------------------------------------

    def _classify(self, h_ratio: float, v_ratio: float) -> str:
        """
        Classify gaze direction using deviation from neutral reference.
        If not yet calibrated, fall back to absolute thresholds as a
        rough starting point.
        """
        if self._calibrated:
            dh = h_ratio - self._neutral_h   # positive = iris moved right = looking right
            dv = v_ratio - self._neutral_v   # positive = iris moved down  = looking down

            # Vertical takes priority over horizontal (down-gaze is safety-critical)
            if dv < -self._v_up_thresh:
                return "up"
            if dv >  self._v_down_thresh:
                return "down"
            if dh < -self._h_thresh:
                return "left"
            if dh >  self._h_thresh:
                return "right"
            return "forward"

        else:
            # Pre-calibration fallback — wide zones to avoid false alerts
            # while calibration is in progress.
            if v_ratio < 0.25:
                return "up"
            if v_ratio > 0.75:
                return "down"
            if h_ratio < 0.30:
                return "left"
            if h_ratio > 0.70:
                return "right"
            return "forward"

    # ------------------------------------------------------------------
    # Direction stabilizer
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, result_dict: dict) -> dict:
        if not result_dict.get('valid', False) or \
                result_dict.get('landmarks') is None:
            if self._last_seen_time and \
                    (time.time() - self._last_seen_time) <= self._hold_after_loss_sec:
                result_dict['gaze_direction'] = self._stable_direction
            else:
                self.not_forward_start_time   = None
                result_dict['gaze_direction'] = None
            return result_dict

        landmarks = result_dict['landmarks']

        # Raw iris ratios — average both eyes
        h_left,  v_left  = compute_gaze_ratios(
            landmarks, LEFT_EYE_CONTOUR,  IRIS_LEFT)
        h_right, v_right = compute_gaze_ratios(
            landmarks, RIGHT_EYE_CONTOUR, IRIS_RIGHT)

        h_raw = (h_left + h_right) / 2.0
        v_raw = (v_left + v_right) / 2.0

        # 5-frame temporal smoothing
        self._ratio_buf.append((h_raw, v_raw))
        h_smooth = float(np.mean([p[0] for p in self._ratio_buf]))
        v_smooth = float(np.mean([p[1] for p in self._ratio_buf]))

        # Optional head-pose correction (disabled by default, scales = 0.0)
        # Signs: yaw > 0 → head turned right → iris ratio shifts left → add back
        #        pitch > 0 → head tilted down → iris ratio shifts down → subtract
        yaw   = result_dict.get('yaw',   0.0)
        pitch = result_dict.get('pitch', 0.0)
        h_corrected = h_smooth + yaw   * self._yaw_scale
        v_corrected = v_smooth - pitch * self._pitch_scale

        # Feed calibration during warm-up period
        if not self._calibrated:
            self._feed_calibration(h_corrected, v_corrected)

        # Classify direction
        direction        = self._classify(h_corrected, v_corrected)
        stable_direction = self._update_stable_direction(direction)

        result_dict['gaze_direction']    = stable_direction
        result_dict['gaze_h_ratio']      = h_corrected   # expose for debugging
        result_dict['gaze_v_ratio']      = v_corrected
        result_dict['gaze_calibrated']   = self._calibrated
        result_dict['gaze_calib_progress'] = self.calibration_progress()
        self._last_seen_time = time.time()

        # Alert timer — only fires after calibration is done
        if self._calibrated and stable_direction != "forward":
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