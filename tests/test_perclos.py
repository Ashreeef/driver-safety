import pytest
from src.fatigue.perclos import PERCLOSTracker

def test_perclos_tracker():
    thresholds = {
        'perclos_window_seconds': 2,
        'perclos_alert_level': 0.15,
        'ear_closure_threshold': 0.05,
        'ear_threshold': 0.20,
        'ear_perclos_ratio': 0.27,
    }
    fps = 5

    # Minimal stub — the test only needs perclos_threshold to return a float
    class _FakeCalibrator:
        calibrated = False
        baseline = None
        @property
        def perclos_threshold(self):
            return thresholds['ear_closure_threshold']   # 0.05

    tracker = PERCLOSTracker(thresholds, _FakeCalibrator(), fps)
    
    # Add 9 open frames (value > closure threshold)
    for _ in range(9):
        res = tracker.update({'valid': True, 'ear': 0.4, 'alerts': []})
        assert res.get('perclos') is None  # Buffer not full yet
        
    # Add 1 closed frame (value < closure threshold) -> buffer = 10 (full)
    res = tracker.update({'valid': True, 'ear': 0.0, 'alerts': []})
    
    # Out of 10 buffer frames, 1 is closed. PERCLOS = 0.10
    assert res['perclos'] == 0.10
    assert len(res['alerts']) == 0  # 0.10 is < 0.15 threshold
    
    # Add 1 more closed frame, pushing an open one out
    res = tracker.update({'valid': True, 'ear': 0.0, 'alerts': []})
    
    # Out of 10 buffer frames, 2 are closed. PERCLOS = 0.20
    assert res['perclos'] == 0.20
    assert 'Drowsiness (PERCLOS)' in res['alerts']
