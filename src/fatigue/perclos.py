import collections

class PERCLOSTracker:
    def __init__(self, thresholds: dict, calibrator, fps: int = 15):
        self.window_seconds  = thresholds.get('perclos_window_seconds', 60)
        self.alert_level     = thresholds.get('perclos_alert_level', 0.15)
        self.calibrator      = calibrator          # ← uses calibrated threshold
        self.buffer_size     = int(self.window_seconds * fps)
        self.ear_buffer      = collections.deque(maxlen=self.buffer_size)

    def update(self, result_dict: dict) -> dict:
        if not result_dict.get('valid', False) or \
           result_dict.get('ear') is None:
            result_dict['perclos'] = None
            return result_dict

        self.ear_buffer.append(result_dict['ear'])

        if len(self.ear_buffer) < self.buffer_size:
            result_dict['perclos'] = None
            return result_dict

        # Use calibrated threshold
        thresh = self.calibrator.perclos_threshold
        closed = sum(1 for e in self.ear_buffer if e < thresh)
        perclos = closed / float(self.buffer_size)

        result_dict['perclos'] = perclos
        if perclos > self.alert_level:
            result_dict['alerts'].append('Drowsiness (PERCLOS)')

        return result_dict
