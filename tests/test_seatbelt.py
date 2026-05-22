import pytest
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compliance.seatbelt.roi import ROIExtractor
from src.fusion.smoothers import EMASmoother, MajorityVoteSmoother

def test_roi_extractor_fallback():
    """Test ROIExtractor fallback when no model is provided or pose fails."""
    extractor = ROIExtractor(method='mediapipe')
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Should return full frame and default bbox if no pose detected
    roi, bbox, kps = extractor.extract(dummy_frame)
    
    assert roi.shape == dummy_frame.shape
    assert bbox == (0, 0, 640, 480)
    assert kps == []

def test_ema_smoother():
    """Test EMASmoother logic."""
    smoother = EMASmoother(config={'alpha': 0.5, 'min_frames': 2})
    
    # First frame
    label, conf = smoother.update(0.8)
    # The new default starting EMA is 0.5. With alpha 0.5, on_prob 0.8:
    # 0.5*0.8 + 0.5*0.5 = 0.65
    assert label == 'WARMING'
    
    # Second frame (now exceeds min_frames)
    label, conf = smoother.update(0.8)
    assert label == 'ON'
    assert conf > 0.5

def test_majority_vote_smoother():
    """Test MajorityVoteSmoother logic."""
    smoother = MajorityVoteSmoother(config={'window_size': 3})
    
    smoother.update(1, 0.9) # ON
    smoother.update(0, 0.8) # OFF
    label, conf = smoother.update(0, 0.7) # OFF
    
    assert label == 0 # Majority is OFF
    assert conf == pytest.approx(0.8) # Average of 0.9, 0.8, 0.7
