import cv2
import numpy as np
from typing import Tuple, List, Optional

class GeometricPrior:
    """
    RANSAC-based geometric verification for seatbelt detection.
    Verifies that edge pixels in the torso ROI form a diagonal line 
    consistent with a real seatbelt trajectory.
    """
    
    def __init__(self, n_iter: int = 30, dist_thresh: float = 4.0, min_inliers: int = 15):
        self.n_iter = n_iter
        self.dist_thresh = dist_thresh
        self.min_inliers = min_inliers

    def _extract_belt_candidates(self, roi: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
        if roi is None or roi.size == 0:
            return np.empty((0, 2), np.float32)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)
        ys, xs = np.nonzero(edges)
        return np.column_stack([xs, ys]).astype(np.float32) if len(xs) else np.empty((0, 2), np.float32)

    def _ransac_line_fit(self, pts: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        if len(pts) < self.min_inliers:
            return None, 0.0
            
        best_line, best_n = None, 0
        for _ in range(self.n_iter):
            idx = np.random.choice(len(pts), 2, replace=False)
            p1, p2 = pts[idx[0]], pts[idx[1]]
            dx = p2[0] - p1[0]
            if abs(dx) < 1e-6:
                continue
            m = (p2[1] - p1[1]) / dx
            b = p1[1] - m * p1[0]
            
            dist = np.abs(m * pts[:, 0] - pts[:, 1] + b) / np.sqrt(m ** 2 + 1)
            n = (dist < self.dist_thresh).sum()
            
            if n > best_n:
                best_n = n
                best_line = np.array([m, b])
                
        ratio = best_n / len(pts) if len(pts) else 0.0
        return (best_line if best_n >= self.min_inliers else None), ratio

    def get_score(self, roi: np.ndarray, keypoints: List, roi_bbox: Tuple, slope_tolerance: float = 0.5) -> float:
        """
        Returns a [0, 1] confidence score that the ROI contains a seatbelt strap.
        """
        if roi is None or roi.size == 0:
            return 0.5
            
        cands = self._extract_belt_candidates(roi)
        if not len(cands):
            return 0.5
            
        line, inlier_ratio = self._ransac_line_fit(cands)
        if line is None:
            return 0.3
            
        # Expected slope based on shoulder-hip alignment if landmarks available
        expected_slope = -1.0 
        if keypoints and len(keypoints) >= 4:
            x1r, y1r = roi_bbox[0], roi_bbox[1]
            # kps: [L-shoulder, R-shoulder, L-hip, R-hip]
            # Use Right shoulder and Left hip as primary diagonal for seatbelt
            rs = (keypoints[1][0] - x1r, keypoints[1][1] - y1r)
            lh = (keypoints[2][0] - x1r, keypoints[2][1] - y1r)
            if abs(lh[0] - rs[0]) > 5:
                expected_slope = (lh[1] - rs[1]) / (lh[0] - rs[0])
                
        slope_err = abs(line[0] - expected_slope)
        slope_score = max(0.0, 1.0 - slope_err / (slope_tolerance * 2))
        
        return float(np.clip(0.5 * inlier_ratio + 0.5 * slope_score, 0.0, 1.0))
