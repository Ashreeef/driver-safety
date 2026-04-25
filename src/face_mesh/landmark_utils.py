import numpy as np

# -----------------------------------------------------------------------
# Landmark index constants
# -----------------------------------------------------------------------

RIGHT_EYE_EAR     = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR      = [362, 385, 387, 263, 373, 380]
IRIS_LEFT         = [468, 469, 470, 471, 472]
IRIS_RIGHT        = [473, 474, 475, 476, 477]
HEAD_POSE_PTS     = [1, 152, 263, 33, 287, 57]

# 4-point MAR: extreme upper/lower lip centers + mouth corners.
# 13  = upper lip center (highest point on upper lip)
# 14  = lower lip center (lowest point on lower lip)
# 78  = left mouth corner
# 308 = right mouth corner
# These four give maximum vertical excursion signal for yawn detection.
MOUTH_MAR_TOP   = 13
MOUTH_MAR_BOTTOM = 14
MOUTH_MAR_LEFT  = 78
MOUTH_MAR_RIGHT = 308

# Full eye contour for gaze bounding box (not EAR points)
LEFT_EYE_CONTOUR  = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]
RIGHT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]


# -----------------------------------------------------------------------
# Landmark array conversion helpers
# -----------------------------------------------------------------------

def get_landmarks_array(face_landmarks) -> np.ndarray:
    """
    Converts MediaPipe normalized landmarks list → np.ndarray (N, 3).
    Returns None if no landmarks detected.
    """
    if not face_landmarks:
        return None
    lm = face_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)


def get_pixel_coords(landmarks_array: np.ndarray,
                     img_w: int, img_h: int) -> np.ndarray:
    """
    Converts normalized (N, 3) array → pixel (N, 2) array.
    Only uses x, y columns; z is discarded.
    """
    if landmarks_array is None:
        return None
    coords = np.zeros((landmarks_array.shape[0], 2), dtype=np.float32)
    coords[:, 0] = landmarks_array[:, 0] * img_w
    coords[:, 1] = landmarks_array[:, 1] * img_h
    return coords


# -----------------------------------------------------------------------
# Landmark Stabilizer — Exponential Moving Average (EMA)
# -----------------------------------------------------------------------

class LandmarkStabilizer:
    """
    Smooths landmark positions across frames using EMA to reduce
    per-frame jitter from MediaPipe's sub-pixel estimation noise.

    Formula (per landmark, per coordinate):
        smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t-1]

    Parameters:
        alpha (float): 0 < alpha <= 1
            - 0.2  → heavy smoothing, slight lag, good for static scenes
            - 0.4  → balanced (recommended default)
            - 0.7  → light smoothing, very responsive
            - 1.0  → no smoothing (passthrough)

    Usage:
        stabilizer = LandmarkStabilizer(alpha=0.4)
        smoothed = stabilizer.smooth(raw_landmarks)

    Reset:
        Call stabilizer.reset() when the face disappears for >1 second
        to avoid snapping back to stale positions when the face reappears.
    """

    def __init__(self, alpha: float = 0.4):
        if not (0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]. Got {alpha}")
        self.alpha  = alpha
        self._prev  = None   # previous smoothed landmarks (N, 3) or None

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply EMA to a (N, 3) landmark array.
        On the first call, returns the input unchanged and stores it.
        Returns None if landmarks is None.
        """
        if landmarks is None:
            return None

        if self._prev is None:
            # First valid frame — no previous state, store and pass through
            self._prev = landmarks.astype(np.float32)
            return self._prev

        # EMA: blend current raw with previous smoothed
        smoothed   = self.alpha * landmarks + (1.0 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed

    def reset(self):
        """
        Clears stored state. Call when face is lost for a sustained period
        so the next detection starts fresh instead of snapping from old position.
        """
        self._prev = None