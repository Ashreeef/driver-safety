import collections
import numpy as np
from src.face_mesh.landmark_utils import RIGHT_EYE_EAR, LEFT_EYE_EAR


def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(landmarks: np.ndarray, eye_indices: list) -> float:
    """Computes EAR for one eye from 6 landmark points."""
    p1, p2, p3, p4, p5, p6 = landmarks[eye_indices]
    v1 = euclidean_dist(p2, p6)
    v2 = euclidean_dist(p3, p5)
    h  = euclidean_dist(p1, p4)
    if h == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h)


# -----------------------------------------------------------------------
# Calibrator — runs during the first N seconds of each session
# -----------------------------------------------------------------------

class EARCalibrator:
    """
    Collects open-eye EAR samples during the calibration window,
    then computes a personal baseline and derived thresholds.

    States:
      CALIBRATING  — collecting samples (first calibration_seconds of driving)
      DONE         — baseline established, thresholds available
    """

    def __init__(self, thresholds: dict, fps: int = 15):
        self.calibration_seconds  = thresholds.get('ear_calibration_seconds', 10)
        coverage_ratio            = thresholds.get('ear_calibration_coverage_ratio', 0.6)
        self.target_frames        = max(30, int(self.calibration_seconds * fps * coverage_ratio))
        self.closure_ratio        = thresholds.get('ear_closure_ratio', 0.75)
        self.perclos_ratio        = thresholds.get('ear_perclos_ratio', 0.27)
        self.fallback_threshold   = thresholds.get('ear_threshold', 0.20)
        self.fallback_perclos     = thresholds.get('ear_closure_threshold', 0.15)
        self.min_sample_ear       = thresholds.get('ear_calibration_min', 0.10)
        self.max_sample_ear       = thresholds.get('ear_calibration_max', 0.65)

        self._samples:  list  = []
        self.calibrated: bool = False
        self.baseline:   float = None   # mean open-eye EAR

    @property
    def alert_threshold(self) -> float:
        """EAR below this → consecutive-frame drowsiness alert."""
        if self.calibrated:
            return self.baseline * self.closure_ratio
        return self.fallback_threshold

    @property
    def perclos_threshold(self) -> float:
        """EAR below this → frame counted as closed for PERCLOS."""
        if self.calibrated:
            return self.baseline * self.perclos_ratio
        return self.fallback_perclos

    def feed(self, ear: float) -> bool:
        """
        Feed one raw EAR sample during calibration.
        Returns True when calibration completes on this call.
        Ignores samples outside a plausible alert range (0.15 – 0.55)
        to avoid collecting closed-eye frames.
        """
        if self.calibrated or ear is None:
            return False
        if self.min_sample_ear <= ear <= self.max_sample_ear:
            self._samples.append(ear)
        if len(self._samples) >= self.target_frames:
            self.baseline = float(np.mean(self._samples))
            self.calibrated = True
            print(
                f"[EARCalibrator] Done. "
                f"samples={len(self._samples)}  "
                f"baseline={self.baseline:.3f}  "
                f"alert_threshold={self.alert_threshold:.3f}  "
                f"perclos_threshold={self.perclos_threshold:.3f}"
            )
            return True
        return False

    def progress(self) -> float:
        """Returns calibration progress 0.0 → 1.0."""
        return min(len(self._samples) / self.target_frames, 1.0)


# -----------------------------------------------------------------------
# EAR Tracker — uses calibrated thresholds + rolling smoothing
# -----------------------------------------------------------------------

class EARTracker:
    """
    Computes smoothed EAR and emits alerts.

    Smoothing: 10-frame rolling mean reduces per-frame jitter.
    Thresholds: supplied by EARCalibrator (or fallback defaults).

    Alerts emitted:
      'Drowsiness (EAR)'       — EAR below alert_threshold for consec_frames
      'Fatigue (EAR Trend)'    — EAR drifting down over 30 frames
    """

    def __init__(self, thresholds: dict, calibrator: EARCalibrator):
        self.calibrator    = calibrator
        self.consec_frames = thresholds.get('ear_consec_frames', 4)
        self.trend_drop    = thresholds.get('ear_trend_drop_threshold', 0.06)
        self.smooth_window = thresholds.get('ear_smooth_frames', 10)

        self._counter      = 0
        self._smooth_buf   = collections.deque(maxlen=self.smooth_window)
        self._trend_buf    = collections.deque(maxlen=30)

    def update(self, result_dict: dict) -> dict:
        if not result_dict.get('valid', False) or \
           result_dict.get('landmarks') is None:
            result_dict['ear'] = None
            self._counter = 0
            return result_dict

        lm = result_dict['landmarks']
        raw_ear = (compute_ear(lm, LEFT_EYE_EAR) +
                   compute_ear(lm, RIGHT_EYE_EAR)) / 2.0

        # Feed calibrator during its window
        self.calibrator.feed(raw_ear)

        # Smooth
        self._smooth_buf.append(raw_ear)
        smoothed = float(np.mean(self._smooth_buf))

        result_dict['ear']     = smoothed
        result_dict['ear_raw'] = raw_ear
        result_dict['ear_calibrated'] = self.calibrator.calibrated
        result_dict['ear_baseline']   = self.calibrator.baseline

        # Consecutive-frame alert (uses calibrated threshold)
        if smoothed < self.calibrator.alert_threshold:
            self._counter += 1
            if self._counter >= self.consec_frames:
                result_dict['alerts'].append('Drowsiness (EAR)')
        else:
            self._counter = 0

        # Trend alert — gradual drift downward
        self._trend_buf.append(smoothed)
        if len(self._trend_buf) == 30:
            first  = np.mean(list(self._trend_buf)[:15])
            second = np.mean(list(self._trend_buf)[15:])
            if (first - second) > self.trend_drop:
                result_dict['alerts'].append('Fatigue (EAR Trend)')

        return result_dict