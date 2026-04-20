import pytest
import numpy as np
from src.fatigue.ear import compute_ear, EARTracker

def test_ear_tracker():
    thresholds = {
        'ear_threshold': 0.20,
        'ear_consec_frames': 2,
        'ear_closure_threshold': 0.05,
        'ear_closure_ratio': 0.75,
        'ear_perclos_ratio': 0.27,
        'ear_smooth_frames': 1,       # disable smoothing in unit test
        'ear_trend_drop_threshold': 0.06,
    }

    class _FakeCalibrator:
        calibrated = False
        baseline = None
        def feed(self, ear):
            return False
        @property
        def alert_threshold(self):
            return thresholds['ear_threshold']   # 0.20

    calibrator = _FakeCalibrator()
    tracker = EARTracker(thresholds, calibrator)
    
    landmarks = np.zeros((478, 3))
    
    # Left eye open mock. P1=(0,0), P4=(100,0). P2/P6 distance=40, P3/P5 distance=40
    landmarks[362] = [0, 0, 0]
    landmarks[263] = [100, 0, 0]
    landmarks[385] = [30, 20, 0]
    landmarks[380] = [30, -20, 0]
    landmarks[387] = [70, 20, 0]
    landmarks[373] = [70, -20, 0]
    
    # Right eye open mock. P1=(200,0), P4=(300,0).
    landmarks[33] = [200, 0, 0]
    landmarks[133] = [300, 0, 0]
    landmarks[160] = [230, 20, 0]
    landmarks[144] = [230, -20, 0]
    landmarks[158] = [270, 20, 0]
    landmarks[153] = [270, -20, 0]
    
    result = {'valid': True, 'landmarks': landmarks, 'alerts': []}
    result = tracker.update(result)
    
    # Open eye EAR -> (40+40)/(2 * 100) = 0.40
    assert result['ear'] == 0.40
    assert len(result['alerts']) == 0
    
    # Closed mock. Compress vertical to 0.
    for idx in [385, 380, 387, 373, 160, 144, 158, 153]:
        landmarks[idx][1] = 0
        
    result = {'valid': True, 'landmarks': landmarks, 'alerts': []}
    result = tracker.update(result)
    
    assert result['ear'] < calibrator.alert_threshold   # closed eye, below threshold
    assert 'Drowsiness (EAR)' not in result['alerts']   # only 1 frame so far, need 2

    result = {'valid': True, 'landmarks': landmarks, 'alerts': []}
    result = tracker.update(result)
    
    assert 'Drowsiness (EAR)' in result['alerts']       # 2nd consecutive frame → alert
