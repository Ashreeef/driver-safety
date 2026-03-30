import cv2
import numpy as np
from src.face_mesh.landmark_utils import HEAD_POSE_PTS

def get_3d_model_points():
    """
    Standard 3D generic facial model mapping for the HEAD_POSE_PTS
    Order: [nose, chin, left eye corner, right eye corner, mouth left, mouth right]
    Based on typical cv2 solvePnP usage for head pose.
    Coordinates are roughly scaled to standard face dimensions.
    """
    return np.array([
        (0.0, 0.0, 0.0),             # Nose tip 1
        (0.0, -330.0, -65.0),        # Chin 152
        (-225.0, 170.0, -135.0),     # Left eye left corner 263 (MediaPipe's perspective)
        (225.0, 170.0, -135.0),      # Right eye right corner 33
        (-150.0, -150.0, -125.0),    # Left Mouth corner 287
        (150.0, -150.0, -125.0)      # Right mouth corner 57
    ], dtype=np.float32)

def estimate_head_pose(landmarks_pixel, img_w, img_h):
    """
    Returns pitch, yaw, roll in degrees from pixel landmarks using solvePnP.
    Requires an array of shape (N, 2) in pixel space.
    """
    if landmarks_pixel is None:
        return 0.0, 0.0, 0.0

    model_points = get_3d_model_points()
    # HEAD_POSE_PTS: [1, 152, 263, 33, 287, 57]
    image_points = landmarks_pixel[HEAD_POSE_PTS]

    # Assume a standard camera intrinsic matrix
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float32
    )

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Get rotational matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Get projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    
    # Decompose projection matrix to get Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
    
    pitch = float(euler_angles[0][0])
    yaw = float(euler_angles[1][0])
    roll = float(euler_angles[2][0])

    # Note: MediaPipe and OpenCV axes might be flipped depending on setup.
    # The returned yaw and pitch should be validated for correct signage later.
    return pitch, yaw, roll
