import cv2
import numpy as np
from src.face_mesh.landmark_utils import HEAD_POSE_PTS


# -----------------------------------------------------------------------
# 3D face model (canonical, scale-independent)
# -----------------------------------------------------------------------

_3D_MODEL = np.array([
    (  0.0,    0.0,   0.0),   # 0 — Nose tip          landmark 1
    (  0.0, -330.0, -65.0),   # 1 — Chin              landmark 152
    (-225.0,  170.0,-135.0),  # 2 — Left eye corner   landmark 263
    ( 225.0,  170.0,-135.0),  # 3 — Right eye corner  landmark 33
    (-150.0, -150.0,-125.0),  # 4 — Left mouth corner landmark 287
    ( 150.0, -150.0,-125.0),  # 5 — Right mouth corner landmark 57
], dtype=np.float32)


def _build_camera_matrix(img_w: int, img_h: int) -> np.ndarray:
    """Approximate pinhole camera matrix assuming standard focal length."""
    f  = float(img_w)
    cx = img_w / 2.0
    cy = img_h / 2.0
    return np.array([[f,  0, cx],
                     [0,  f, cy],
                     [0,  0,  1]], dtype=np.float32)


# -----------------------------------------------------------------------
# Method A — solvePnP
# -----------------------------------------------------------------------

def _pose_from_solvepnp(pixel_coords: np.ndarray,
                         img_w: int, img_h: int):
    """
    Returns (pitch, yaw, roll) in degrees or None on failure.
    """
    if pixel_coords is None:
        return None

    image_points  = pixel_coords[HEAD_POSE_PTS].astype(np.float32)
    camera_matrix = _build_camera_matrix(img_w, img_h)
    dist_coeffs   = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        _3D_MODEL,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    proj  = np.hstack((R, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)

    pitch = float(euler[0][0])
    yaw   = float(euler[1][0])
    roll  = float(euler[2][0])

    # Sanity check — reject physically implausible values
    if abs(yaw) > 90 or abs(pitch) > 90:
        return None

    return pitch, yaw, roll


# -----------------------------------------------------------------------
# Method B — MediaPipe transformation matrix (fast fallback)
# -----------------------------------------------------------------------

def _pose_from_transform_matrix(transform_matrix) -> tuple:
    """
    Extract Euler angles from MediaPipe's facial_transformation_matrix.
    This is computed internally by MediaPipe at no extra cost when
    output_facial_transformation_matrixes=True is set.

    Returns (pitch, yaw, roll) in degrees. Never fails.
    """
    if transform_matrix is None:
        return 0.0, 0.0, 0.0

    mat = np.array(transform_matrix.data, dtype=np.float32).reshape(4, 4)
    R   = mat[:3, :3]

    pitch = np.degrees(np.arctan2(-R[2, 1],  R[2, 2]))
    yaw   = np.degrees(np.arctan2( R[2, 0],
                                   np.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    roll  = np.degrees(np.arctan2(-R[1, 0],  R[0, 0]))

    return float(pitch), float(yaw), float(roll)


# -----------------------------------------------------------------------
# Public function — tries solvePnP, falls back to transform matrix
# -----------------------------------------------------------------------

def estimate_head_pose(pixel_coords: np.ndarray,
                       img_w: int, img_h: int,
                       transform_matrix=None) -> tuple:
    """
    Returns (pitch, yaw, roll, method) in degrees.

    Strategy:
      1. Try solvePnP. If it succeeds and values are plausible → use it.
      2. If solvePnP fails or returns implausible values:
           → use MediaPipe transformation matrix if available.
      3. If neither works → return (0, 0, 0, 'none').

    Parameters:
        pixel_coords     : (N, 2) array of pixel-space landmarks
        img_w, img_h     : frame dimensions
        transform_matrix : mediapipe facial_transformation_matrix or None

    Returns:
        (pitch, yaw, roll, method)
        method is one of: 'solvepnp', 'transform_matrix', 'none'
    """
    # Attempt Method A
    result_pnp = _pose_from_solvepnp(pixel_coords, img_w, img_h)
    if result_pnp is not None:
        pitch, yaw, roll = result_pnp
        return pitch, yaw, roll, 'solvepnp'

    # Fallback to Method B
    if transform_matrix is not None:
        pitch, yaw, roll = _pose_from_transform_matrix(transform_matrix)
        return pitch, yaw, roll, 'transform_matrix'

    # Nothing worked
    return 0.0, 0.0, 0.0, 'none'