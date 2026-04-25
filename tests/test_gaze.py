import time
import numpy as np
import pytest
from src.gaze.gaze_estimator import GazeEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _thresholds(**overrides):
    base = {
        'gaze_calib_frames':              10,   # short for tests
        'gaze_calib_max_head_yaw':        10.0,
        'gaze_calib_max_head_pitch':       8.0,
        'gaze_head_yaw_threshold':        20.0,
        'gaze_head_pitch_down_threshold': 15.0,
        'gaze_head_pitch_up_threshold':   10.0,
        'gaze_h_threshold':               0.14,
        'gaze_v_up_threshold':            0.10,
        'gaze_v_down_threshold':          0.12,
        'gaze_vertical_invert':           False,
        'gaze_yaw_correction_scale':      0.0,
        'gaze_pitch_correction_scale':    0.0,
        'gaze_smoothing_frames':          1,    # no smoothing — deterministic tests
        'gaze_confirm_frames':            1,    # instant confirmation — tests direction logic
        'gaze_alert_seconds':             9999, # suppress alert timer in most tests
        'gaze_hold_after_loss_sec':       0.0,
    }
    base.update(overrides)
    return base


def _make_result(yaw=0.0, pitch=0.0, valid=True):
    """Minimal result_dict with landmarks=None (we test head-pose path mostly)."""
    return {
        'valid':      valid,
        'landmarks':  None,
        'yaw':        yaw,
        'pitch':      pitch,
        'alerts':     [],
        'gaze_direction':      None,
        'gaze_h_ratio':        None,
        'gaze_v_ratio':        None,
        'gaze_calibrated':     False,
        'gaze_calib_progress': 0.0,
    }


# ---------------------------------------------------------------------------
# Head-pose primary classification
# ---------------------------------------------------------------------------

class TestHeadDirection:

    def test_forward_within_thresholds(self):
        g = GazeEstimator(_thresholds())
        assert g._head_direction(0.0, 0.0)   == "forward"
        assert g._head_direction(10.0, 5.0)  == "forward"
        assert g._head_direction(-10.0, -5.0) == "forward"

    def test_right_when_yaw_exceeds(self):
        g = GazeEstimator(_thresholds())
        assert g._head_direction(20.1, 0.0) == "right"
        assert g._head_direction(45.0, 0.0) == "right"

    def test_left_when_yaw_negative(self):
        g = GazeEstimator(_thresholds())
        assert g._head_direction(-20.1, 0.0) == "left"

    def test_down_when_pitch_exceeds(self):
        g = GazeEstimator(_thresholds())
        assert g._head_direction(0.0, 15.1) == "down"

    def test_up_when_pitch_negative(self):
        g = GazeEstimator(_thresholds())
        assert g._head_direction(0.0, -10.1) == "up"

    def test_yaw_beats_pitch_in_priority(self):
        # Both yaw and pitch exceed — yaw is checked first
        g = GazeEstimator(_thresholds())
        assert g._head_direction(25.0, 20.0) == "right"


# ---------------------------------------------------------------------------
# Calibration gating
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_calibration_rejects_high_yaw_frames(self):
        g = GazeEstimator(_thresholds(gaze_calib_frames=5))
        # Feed 5 frames with high yaw — all should be rejected
        for _ in range(5):
            g._feed_calibration(0.5, 0.5, head_yaw=15.0, head_pitch=0.0)
        assert not g._calibrated
        assert len(g._calib_h_buf) == 0

    def test_calibration_rejects_high_pitch_frames(self):
        g = GazeEstimator(_thresholds(gaze_calib_frames=5))
        for _ in range(5):
            g._feed_calibration(0.5, 0.5, head_yaw=0.0, head_pitch=10.0)
        assert not g._calibrated

    def test_calibration_completes_on_forward_frames(self):
        g = GazeEstimator(_thresholds(gaze_calib_frames=5))
        for _ in range(5):
            g._feed_calibration(0.48, 0.52, head_yaw=2.0, head_pitch=1.0)
        assert g._calibrated
        assert abs(g._neutral_h - 0.48) < 0.01
        assert abs(g._neutral_v - 0.52) < 0.01

    def test_calibration_progress(self):
        g = GazeEstimator(_thresholds(gaze_calib_frames=10))
        for i in range(5):
            g._feed_calibration(0.5, 0.5, head_yaw=0.0, head_pitch=0.0)
        assert abs(g.calibration_progress() - 0.5) < 0.01

    def test_reset_clears_state(self):
        g = GazeEstimator(_thresholds(gaze_calib_frames=5))
        for _ in range(5):
            g._feed_calibration(0.5, 0.5, head_yaw=0.0, head_pitch=0.0)
        assert g._calibrated
        g.reset_calibration()
        assert not g._calibrated
        assert g._neutral_h is None
        assert g._stable_direction == "forward"
        assert g.not_forward_start_time is None


# ---------------------------------------------------------------------------
# Iris classification (post-calibration)
# ---------------------------------------------------------------------------

class TestIrisDirection:

    def _calibrated_estimator(self, neutral_h=0.50, neutral_v=0.50):
        g = GazeEstimator(_thresholds())
        g._calibrated = True
        g._neutral_h  = neutral_h
        g._neutral_v  = neutral_v
        return g

    def test_forward_at_neutral(self):
        g = self._calibrated_estimator()
        assert g._iris_direction(0.50, 0.50) == "forward"

    def test_right_on_positive_dh(self):
        g = self._calibrated_estimator()
        # dh = 0.50 + 0.15 - 0.50 = 0.15 > 0.14 threshold
        assert g._iris_direction(0.65, 0.50) == "right"

    def test_left_on_negative_dh(self):
        g = self._calibrated_estimator()
        assert g._iris_direction(0.35, 0.50) == "left"

    def test_up_on_positive_dv_invert_false(self):
        # invert=False: positive dv_raw → "up"
        g = self._calibrated_estimator(neutral_v=0.50)
        # dv_raw = 0.62 - 0.50 = 0.12 > 0.10 threshold → "up"
        assert g._iris_direction(0.50, 0.62) == "up"

    def test_down_on_negative_dv_invert_false(self):
        g = self._calibrated_estimator(neutral_v=0.50)
        # dv_raw = 0.37 - 0.50 = -0.13 < -0.12 → "down"
        assert g._iris_direction(0.50, 0.37) == "down"

    def test_within_dead_zone_returns_forward(self):
        g = self._calibrated_estimator()
        # Small deviation — inside threshold
        assert g._iris_direction(0.55, 0.55) == "forward"


# ---------------------------------------------------------------------------
# Head-pose primary gates iris (regression test for projection artefact fix)
# ---------------------------------------------------------------------------

class TestFusedClassify:

    def test_head_primary_overrides_iris(self):
        # Even if iris says "left", head yaw > threshold → "right"
        g = GazeEstimator(_thresholds())
        g._calibrated = True
        g._neutral_h  = 0.50
        g._neutral_v  = 0.50
        # iris deviation says left (h=0.30), but head yaw=25 says right
        result = g._classify(0.30, 0.50, head_yaw=25.0, head_pitch=0.0)
        assert result == "right"

    def test_iris_used_when_head_forward(self):
        g = GazeEstimator(_thresholds())
        g._calibrated = True
        g._neutral_h  = 0.50
        g._neutral_v  = 0.50
        # head is near-forward, iris deviation dominates
        result = g._classify(0.65, 0.50, head_yaw=5.0, head_pitch=2.0)
        assert result == "right"


# ---------------------------------------------------------------------------
# Alert timer
# ---------------------------------------------------------------------------

class TestAlertTimer:

    def test_alert_fires_after_duration(self):
        g = GazeEstimator(_thresholds(
            gaze_alert_seconds=0.05,
            gaze_confirm_frames=1,
        ))
        g._calibrated = True
        # Manually set stable direction to non-forward
        g._stable_direction = "left"
        g.not_forward_start_time = time.time() - 0.1   # already past threshold

        result = {'alerts': [], 'valid': True}
        g._update_alert(result, "left")
        assert 'Distraction (Gaze)' in result['alerts']

    def test_alert_resets_on_forward(self):
        g = GazeEstimator(_thresholds(gaze_alert_seconds=0.05))
        g._calibrated = True
        g.not_forward_start_time = time.time() - 1.0

        result = {'alerts': []}
        g._update_alert(result, "forward")
        assert g.not_forward_start_time is None
        assert 'Distraction (Gaze)' not in result['alerts']

    def test_no_alert_before_calibration(self):
        g = GazeEstimator(_thresholds(gaze_alert_seconds=0.0))
        g._calibrated = False
        g.not_forward_start_time = time.time() - 1.0

        result = {'alerts': []}
        g._update_alert(result, "left")
        assert 'Distraction (Gaze)' not in result['alerts']


# ---------------------------------------------------------------------------
# Invalid frame handling
# ---------------------------------------------------------------------------

class TestInvalidFrame:

    def test_invalid_frame_sets_gaze_keys(self):
        g = GazeEstimator(_thresholds())
        result = _make_result(valid=False)
        result = g.update(result)
        assert 'gaze_calibrated' in result
        assert 'gaze_calib_progress' in result
        assert result['gaze_direction'] is None

    def test_valid_false_does_not_advance_calibration(self):
        g = GazeEstimator(_thresholds(gaze_calib_frames=5))
        for _ in range(5):
            result = _make_result(valid=False)
            g.update(result)
        assert not g._calibrated
