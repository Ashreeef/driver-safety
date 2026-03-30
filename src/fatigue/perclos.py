import collections

class PERCLOSTracker:
    def __init__(self, thresholds: dict, fps: int = 15):
        """
        Initializes PERCLOS tracker.
        Requires estimated `fps` to determine the raw buffer size.
        Target edge hardware is exactly 15 FPS for the baseline.
        """
        self.window_seconds = thresholds.get('perclos_window_seconds', 60)
        self.alert_level = thresholds.get('perclos_alert_level', 0.15)
        self.closure_threshold = thresholds.get('ear_closure_threshold', 0.05)
        
        self.buffer_size = int(self.window_seconds * fps)
        self.ear_buffer = collections.deque(maxlen=self.buffer_size)

    def update(self, result_dict: dict) -> dict:
        """
        Calculates PERCLOS over the set window.
        Relies on `result_dict['ear']` already being computed by EARTracker.
        """
        if not result_dict.get('valid', False) or result_dict.get('ear') is None:
            # We don't record invalid frames to the buffer to avoid diluting the ratio
            # with failed detection frames
            if len(self.ear_buffer) < self.buffer_size:
                result_dict['perclos'] = None
            return result_dict

        ear = result_dict['ear']
        self.ear_buffer.append(ear)

        # "Return None until buffer is full"
        if len(self.ear_buffer) < self.buffer_size:
            result_dict['perclos'] = None
            return result_dict

        # Calculate PERCLOS (percentage of eye closure time)
        closed_frames = sum(1 for e in self.ear_buffer if e < self.closure_threshold)
        perclos = closed_frames / float(self.buffer_size)
        
        result_dict['perclos'] = float(perclos)

        if perclos > self.alert_level:
            result_dict['alerts'].append('Drowsiness (PERCLOS)')

        return result_dict
