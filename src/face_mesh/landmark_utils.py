import numpy as np

# Landmark Indices from requirements
RIGHT_EYE_EAR  = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR   = [362, 385, 387, 263, 373, 380]
IRIS_LEFT      = [468, 469, 470, 471, 472]
IRIS_RIGHT     = [473, 474, 475, 476, 477]
HEAD_POSE_PTS  = [1, 152, 263, 33, 287, 57]  # nose, chin, eye corners, mouth corners

# Assumed outer lip indices for MAR based on standard MediaPipe layout
# WILL VERIFY THESE INDICES ON A REAL VIDEO
MOUTH_MAR      = [78, 82, 312, 308, 317, 87] 

# dedicated eye contour for Gaze Box
LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def get_landmarks_array(face_landmarks) -> np.ndarray:
    """
    Converts MediaPipe normalized landmarks list into a NumPy array of shape (N, 3).
    Takes the first detected face's landmarks.
    """
    if not face_landmarks:
        return None
    # Use first detected face
    landmarks = face_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

def get_pixel_coords(landmarks_array: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Converts a (N, 3) or (N, 2) array of normalized [x,y,...] coordinates to pixel [x, y] coordinates.
    """
    if landmarks_array is None:
        return None
    
    pixel_coords = np.zeros((landmarks_array.shape[0], 2), dtype=np.float32)
    pixel_coords[:, 0] = landmarks_array[:, 0] * img_w
    pixel_coords[:, 1] = landmarks_array[:, 1] * img_h
    return pixel_coords
