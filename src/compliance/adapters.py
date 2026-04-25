"""
Thin bridges that write compliance module outputs into the shared result_dict.

Every function receives the module's native output and the current result_dict,
writes the contract keys, and returns result_dict.  Never instantiate a new dict
inside — always mutate and return the one that was passed in.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.compliance.smoking.detector import SmokingFrameResult
    from src.compliance.phone import PhoneFrameResult


def seatbelt_to_result(scene_dict: dict, result_dict: dict) -> dict:
    """
    Bridge Pipeline1 / Pipeline2 / Pipeline3 scene_dict → result_dict.

    All three pipelines share the 'label' key ('ON' / 'OFF' / 'WARMING').
    'WARMING' is treated as not-yet-detected (False) until the smoother
    has enough frames to make a stable prediction.
    """
    result_dict['seatbelt_detected'] = (scene_dict.get('label') == 'ON')
    return result_dict


def smoking_to_result(sm_res: SmokingFrameResult, result_dict: dict) -> dict:
    """
    Bridge SmokingFrameResult → result_dict.

    Mapping:
      alert_active        → smoking_detected
      temporal_conf       → smoking_proximity_sec (0–1 fraction as proxy;
                            exact seconds not tracked in Imen's model)
      'right_wrist' in lm → smoking_hand_visible
      smoking_wrist_ok    → True (Imen's model uses elbow-angle guard
                            instead of the wrist-height check from the old
                            behavioral approach)
    """
    lm = sm_res.landmarks or {}
    result_dict['smoking_detected']      = sm_res.alert_active
    result_dict['smoking_proximity_sec'] = float(sm_res.temporal_conf)
    result_dict['smoking_hand_visible']  = 'right_wrist' in lm
    result_dict['smoking_wrist_ok']      = True
    return result_dict


def phone_to_result(ph_res: PhoneFrameResult, result_dict: dict) -> dict:
    """
    Bridge PhoneFrameResult → result_dict.

    Mapping:
      alert_triggered → phone_detected
    """
    result_dict['phone_detected'] = ph_res.alert_triggered
    return result_dict
