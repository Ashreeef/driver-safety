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
    iris_center = np.mean(landmarks[iris_indices], axis=0)
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
    Fused head-pose + iris gaze estimator.

    Signal hierarchy:
      1. HEAD POSE PRIMARY — when |yaw| or |pitch| exceeds a threshold, gaze
         direction is read directly from head angles. Head pose is available
         immediately (no calibration), is not affected by iris projection
         artefacts, and is reliable for large deviations.
      2. IRIS DEVIATION — when head is near-forward, iris position relative to
         a calibrated neutral detects eye-only shifts (e.g. looking at mirrors
         without moving the head). This signal is gated to the region where
         the eye bounding box is approximately undistorted.

    Calibration:
      The first gaze_calib_frames valid frames where the head is also
      near-forward (|yaw| < calib_max_head_yaw, |pitch| < calib_max_head_pitch)
      establish the personal neutral iris reference. Gating on head angle
      prevents a biased neutral from a session started mid-glance.

    Alert logic:
      Direction must be stable for gaze_confirm_frames consecutive frames
      before the alert clock starts. Alert fires after gaze_alert_seconds of
      sustained non-forward gaze. Both thresholds are intentionally conservative
      to avoid alerting on normal mirror checks (< 2s) or brief glances.
    """

    def __init__(self, thresholds: dict):
        self.alert_seconds        = thresholds.get('gaze_alert_seconds', 2.5)
        self._hold_after_loss_sec = thresholds.get('gaze_hold_after_loss_sec', 0.4)
        self._confirm_frames      = thresholds.get('gaze_confirm_frames', 8)
        self._calib_frames        = int(thresholds.get('gaze_calib_frames', 40))

        # ---- Head-pose primary thresholds (degrees) ----
        # When the head exceeds these angles, the iris bounding box is too
        # distorted to trust — head pose drives the direction instead.
        self._head_yaw_thresh        = thresholds.get('gaze_head_yaw_threshold',        20.0)
        self._head_pitch_down_thresh = thresholds.get('gaze_head_pitch_down_threshold', 15.0)
        self._head_pitch_up_thresh   = thresholds.get('gaze_head_pitch_up_threshold',   10.0)

        # ---- Calibration quality gate ----
        # Only accept a frame as a neutral sample when the head is also near-forward.
        self._calib_max_yaw   = thresholds.get('gaze_calib_max_head_yaw',   10.0)
        self._calib_max_pitch = thresholds.get('gaze_calib_max_head_pitch',   8.0)

        # ---- Iris deviation thresholds ----
        # Active only when head is within primary bounds and calibration is done.
        # Horizontal: symmetric left/right.
        # Vertical: separate thresholds because eyelid covers upward gaze more.
        self._h_thresh      = thresholds.get('gaze_h_threshold',      0.14)
        self._v_up_thresh   = thresholds.get('gaze_v_up_threshold',   0.10)
        self._v_down_thresh = thresholds.get('gaze_v_down_threshold', 0.12)

        # gaze_vertical_invert=False → this camera: small v_ratio means looking down.
        # gaze_vertical_invert=True  → standard camera: small v_ratio means looking up.
        self._invert_vertical = thresholds.get('gaze_vertical_invert', True)

        # Head-pose correction for iris (keep 0.0 until iris-only is verified)
        self._yaw_scale   = thresholds.get('gaze_yaw_correction_scale',   0.0)
        self._pitch_scale = thresholds.get('gaze_pitch_correction_scale', 0.0)

        # ---- Calibration state ----
        self._neutral_h   = None
        self._neutral_v   = None
        self._calib_h_buf = []
        self._calib_v_buf = []
        self._calibrated  = False

        # ---- Iris smoothing buffer ----
        smooth_frames = int(thresholds.get('gaze_smoothing_frames', 5))
        self._ratio_buf = collections.deque(maxlen=max(1, smooth_frames))

        # ---- Direction stabiliser ----
        self._stable_direction    = "forward"
        self._candidate_direction = None
        self._candidate_count     = 0

        # ---- Alert timer ----
        self.not_forward_start_time = None
        self._last_seen_time        = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _feed_calibration(self, h: float, v: float,
                           head_yaw: float, head_pitch: float):
        """
        Collect neutral-gaze iris samples only when the head is near-forward.
        Frames where the head is already turned are rejected so that the
        computed neutral is not biased toward one direction.
        """
        if abs(head_yaw) > self._calib_max_yaw:
            return
        if abs(head_pitch) > self._calib_max_pitch:
            return
        self._calib_h_buf.append(h)
        self._calib_v_buf.append(v)
        if len(self._calib_h_buf) >= self._calib_frames:
            self._neutral_h  = float(np.median(self._calib_h_buf))
            self._neutral_v  = float(np.median(self._calib_v_buf))
            self._calibrated = True
            print(
                f"[GazeEstimator] Calibrated. "
                f"neutral_h={self._neutral_h:.3f}  "
                f"neutral_v={self._neutral_v:.3f}"
            )

    def calibration_progress(self) -> float:
        return min(len(self._calib_h_buf) / self._calib_frames, 1.0)

    def reset_calibration(self):
        """Force re-calibration (driver change or camera repositioned)."""
        self._calib_h_buf.clear()
        self._calib_v_buf.clear()
        self._calibrated  = False
        self._neutral_h   = None
        self._neutral_v   = None
        self._ratio_buf.clear()
        self._stable_direction    = "forward"
        self._candidate_direction = None
        self._candidate_count     = 0
        self.not_forward_start_time = None

    # ------------------------------------------------------------------
    # Direction classification
    # ------------------------------------------------------------------

    def _head_direction(self, yaw: float, pitch: float) -> str:
        """
        Classify gaze from head pose alone.
        Returns "forward" when head is within primary thresholds.
        """
        if yaw > self._head_yaw_thresh:
            return "right"
        if yaw < -self._head_yaw_thresh:
            return "left"
        if pitch > self._head_pitch_down_thresh:
            return "down"
        if pitch < -self._head_pitch_up_thresh:
            return "up"
        return "forward"

    def _iris_direction(self, h_ratio: float, v_ratio: float) -> str:
        """
        Classify gaze from iris deviation from calibrated neutral.
        Only called when head is within primary bounds and calibration done.
        """
        dh     = h_ratio - self._neutral_h
        dv_raw = v_ratio - self._neutral_v
        # Vertical sign convention:
        #   invert=False (this camera): positive dv_raw → looking up
        #   invert=True  (standard)  : positive dv_raw → looking down → need to negate
        dv = -dv_raw if self._invert_vertical else dv_raw

        if dv > self._v_up_thresh:
            return "up"
        if dv < -self._v_down_thresh:
            return "down"
        if dh < -self._h_thresh:
            return "left"
        if dh > self._h_thresh:
            return "right"
        return "forward"

    def _classify(self, h_ratio: float, v_ratio: float,
                  head_yaw: float, head_pitch: float) -> str:
        """
        Fused classification.

        Head pose is checked first. When the head is near-forward, iris
        deviation from neutral is used. This ordering ensures that iris
        is never applied in the zone where bounding box projection artefacts
        make it unreliable.
        """
        head_dir = self._head_direction(head_yaw, head_pitch)
        if head_dir != "forward":
            return head_dir

        if self._calibrated:
            return self._iris_direction(h_ratio, v_ratio)

        # Pre-calibration fallback — wide absolute zones, consistent with
        # calibrated convention for the current invert_vertical setting.
        if self._invert_vertical:
            if v_ratio < 0.25:
                return "up"
            if v_ratio > 0.75:
                return "down"
        else:
            if v_ratio < 0.25:
                return "down"
            if v_ratio > 0.75:
                return "up"
        if h_ratio < 0.30:
            return "left"
        if h_ratio > 0.70:
            return "right"
        return "forward"

    # ------------------------------------------------------------------
    # Direction stabiliser
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
            result_dict['gaze_calibrated']     = self._calibrated
            result_dict['gaze_calib_progress'] = self.calibration_progress()
            return result_dict

        landmarks  = result_dict['landmarks']
        head_yaw   = result_dict.get('yaw',   0.0)
        head_pitch = result_dict.get('pitch', 0.0)

        # Raw iris ratios — average both eyes for lower variance
        h_left,  v_left  = compute_gaze_ratios(
            landmarks, LEFT_EYE_CONTOUR,  IRIS_LEFT)
        h_right, v_right = compute_gaze_ratios(
            landmarks, RIGHT_EYE_CONTOUR, IRIS_RIGHT)

        h_raw = (h_left + h_right) / 2.0
        v_raw = (v_left + v_right) / 2.0

        # Temporal smoothing — reduces frame-to-frame jitter from MediaPipe noise
        self._ratio_buf.append((h_raw, v_raw))
        h_smooth = float(np.mean([p[0] for p in self._ratio_buf]))
        v_smooth = float(np.mean([p[1] for p in self._ratio_buf]))

        # Head-pose correction (disabled by default — enable after iris-only verified)
        h_corrected = h_smooth + head_yaw   * self._yaw_scale
        v_corrected = v_smooth - head_pitch * self._pitch_scale

        # Feed calibration with head-gated sample collection
        if not self._calibrated:
            self._feed_calibration(h_corrected, v_corrected, head_yaw, head_pitch)

        # Classify using fused head + iris signal
        direction        = self._classify(h_corrected, v_corrected, head_yaw, head_pitch)
        stable_direction = self._update_stable_direction(direction)

        result_dict['gaze_direction']      = stable_direction
        result_dict['gaze_h_ratio']        = h_corrected
        result_dict['gaze_v_ratio']        = v_corrected
        result_dict['gaze_calibrated']     = self._calibrated
        result_dict['gaze_calib_progress'] = self.calibration_progress()
        self._last_seen_time = time.time()

        self._update_alert(result_dict, stable_direction)
        return result_dict

    def _update_alert(self, result_dict: dict, stable_direction: str):
        """Drive the distraction alert timer. Extracted for testability."""
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
