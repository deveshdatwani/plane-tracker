import numpy as np
import cv2
import logging
from scipy.optimize import linear_sum_assignment
from src.lib.drawing import draw_track

logger = logging.getLogger(__name__)


def extract_feature_vector(frame: np.ndarray, mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from a detection for Re-ID.
    Combines color histogram and shape features.
    
    Args:
        frame: BGR image
        mask: Binary mask for the detection
        bbox: [x1, y1, x2, y2] bounding box
    
    Returns:
        1D feature vector (normalized)
    """
    features = []
    
    # 1. Color histogram (HSV) - 32 bins per channel = 96 dims
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
    
    for i, (bins, range_max) in enumerate([(32, 180), (32, 256), (32, 256)]):
        hist = cv2.calcHist([hsv], [i], mask_uint8, [bins], [0, range_max])
        hist = hist.flatten()
        if hist.sum() > 0:
            hist = hist / hist.sum()  # Normalize
        features.extend(hist)
    
    # 2. Shape features (4 dims)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    area = mask.sum()
    bbox_area = w * h if w * h > 0 else 1
    
    # Aspect ratio
    aspect = w / h if h > 0 else 1
    # Solidity (area / convex hull area)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 1.0
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1
    
    # Extent (area / bbox area)
    extent = area / bbox_area
    
    # Normalized area (relative to frame size)
    frame_area = frame.shape[0] * frame.shape[1]
    norm_area = area / frame_area
    
    features.extend([aspect, solidity, extent, norm_area])
    
    feature_vec = np.array(features, dtype=np.float32)
    
    # L2 normalize full vector
    norm = np.linalg.norm(feature_vec)
    if norm > 0:
        feature_vec = feature_vec / norm
    
    return feature_vec


def extract_mask_keypoints(mask: np.ndarray, max_points: int = 50) -> np.ndarray:
    """
    Extract keypoints from a segmentation mask using multiple methods:
    1. Contour corners (Douglas-Peucker approximation)
    2. Shi-Tomasi corners inside mask
    3. Convex hull vertices
    
    Args:
        mask: Binary mask (H x W boolean or uint8)
        max_points: Maximum number of keypoints to return
    
    Returns:
        Nx2 array of (x, y) keypoint coordinates
    """
    mask_uint8 = (mask.astype(np.uint8) * 255) if mask.dtype == bool else mask
    
    keypoints = []
    
    # 1. Contour-based keypoints (corners of polygon approximation)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Douglas-Peucker polygon approximation - extracts corner points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        for pt in approx:
            keypoints.append(pt[0])
        
        # Convex hull vertices
        hull = cv2.convexHull(contour)
        for pt in hull:
            keypoints.append(pt[0])
    
    # 2. Shi-Tomasi corner detection inside mask
    corners = cv2.goodFeaturesToTrack(
        mask_uint8, 
        maxCorners=max_points // 2,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask_uint8
    )
    if corners is not None:
        for corner in corners:
            keypoints.append(corner[0].astype(int))
    
    # 3. Centroid
    M = cv2.moments(mask_uint8)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        keypoints.append([cx, cy])
    
    # Remove duplicates (within 5px tolerance)
    if keypoints:
        keypoints = np.array(keypoints, dtype=np.float32)
        unique_kps = [keypoints[0]]
        for kp in keypoints[1:]:
            dists = np.linalg.norm(np.array(unique_kps) - kp, axis=1)
            if np.min(dists) > 5:
                unique_kps.append(kp)
        keypoints = np.array(unique_kps[:max_points], dtype=np.float32)
    else:
        keypoints = np.array([], dtype=np.float32).reshape(0, 2)
    
    return keypoints


class Tracklet:
    def __init__(self, track_id, bbox, mask, max_keypoints=20):
        self.id = track_id
        self.miss_count = 0
        self.mask = mask
        self.age = 0
        self.hits = 0
        self.max_keypoints = max_keypoints  # Configurable max keypoints
        self.dim_x, self.dim_z = 8, 4
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 1e-5
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(4): self.F[i, i + 4] = 1
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[:4, :4] = np.eye(4)
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 0.01
        self.R = np.eye(self.dim_z, dtype=np.float32) * 1e-1
        self.bbox = np.array(bbox)
        
        # Keypoint tracking
        self.keypoints = extract_mask_keypoints(mask, max_points=max_keypoints)  # Nx2 array
        self.keypoint_ids = np.arange(len(self.keypoints))  # Unique ID per keypoint
        self._next_kp_id = len(self.keypoints)
        self._prev_gray = None  # Store grayscale for optical flow
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.miss_count += 1
        cx, cy, w, h = self.x[:4].flatten()
        self.bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        return self.bbox
    
    def update(self, new_bbox, new_mask, gray_frame=None):
        self.miss_count = 0
        self.hits += 1
        x1, y1, x2, y2 = new_bbox
        z = np.array([[x1 + (x2-x1)/2], [y1 + (y2-y1)/2], [x2-x1], [y2-y1]], dtype=np.float32)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P
        self.bbox = np.array(new_bbox)
        self.mask = new_mask
        
        # Track keypoints using optical flow if we have previous frame
        if gray_frame is not None and self._prev_gray is not None and len(self.keypoints) > 0:
            # Lucas-Kanade optical flow
            old_pts = self.keypoints.reshape(-1, 1, 2).astype(np.float32)
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray_frame, old_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            if new_pts is not None:
                # Filter by status and keep only points inside new mask
                status = status.flatten()
                tracked_kps = []
                tracked_ids = []
                
                for i, (pt, st) in enumerate(zip(new_pts, status)):
                    if st == 1:
                        px, py = int(pt[0, 0]), int(pt[0, 1])
                        # Check if point is inside new mask
                        if 0 <= py < new_mask.shape[0] and 0 <= px < new_mask.shape[1]:
                            if new_mask[py, px]:
                                tracked_kps.append(pt[0])
                                tracked_ids.append(self.keypoint_ids[i])
                
                # Extract new keypoints from mask
                new_kps = extract_mask_keypoints(new_mask, max_points=self.max_keypoints)
                
                # Merge: keep tracked points + add new ones that are far from existing
                if tracked_kps:
                    self.keypoints = np.array(tracked_kps, dtype=np.float32)
                    self.keypoint_ids = np.array(tracked_ids, dtype=np.int32)
                    
                    # Add new keypoints that are far from tracked ones
                    for new_kp in new_kps:
                        if len(self.keypoints) >= self.max_keypoints:
                            break
                        dists = np.linalg.norm(self.keypoints - new_kp, axis=1)
                        if np.min(dists) > 15:  # At least 15px away
                            self.keypoints = np.vstack([self.keypoints, new_kp])
                            self.keypoint_ids = np.append(self.keypoint_ids, self._next_kp_id)
                            self._next_kp_id += 1
                else:
                    # Lost all keypoints, re-extract
                    self.keypoints = new_kps
                    self.keypoint_ids = np.arange(self._next_kp_id, self._next_kp_id + len(new_kps))
                    self._next_kp_id += len(new_kps)
        else:
            # First frame or no previous gray - just extract from mask
            self.keypoints = extract_mask_keypoints(new_mask, max_points=self.max_keypoints)
            self.keypoint_ids = np.arange(self._next_kp_id, self._next_kp_id + len(self.keypoints))
            self._next_kp_id += len(self.keypoints)
        
        # Store current gray for next frame
        self._prev_gray = gray_frame
    
    def to_dict(self):
        return {
            "track_id": self.id,
            "bbox": self.bbox.tolist(),
            "state": self.x.flatten().tolist(),
            "age": self.age,
            "hits": self.hits,
            "miss_count": self.miss_count,
        }

class PlaneTracker:
    def __init__(self, iou_threshold=None, max_age=None, min_hits=None):
        from src.config import get_config
        cfg = get_config()
        tracker_cfg = cfg.get("tracker", {})
        reid_cfg = cfg.get("reid", {})
        
        self.trackers = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold if iou_threshold is not None else tracker_cfg.get("iou_threshold", 0.2)
        self.max_age = max_age if max_age is not None else tracker_cfg.get("max_age", 30)
        self.min_hits = min_hits if min_hits is not None else tracker_cfg.get("min_hits", 10)
        self.max_keypoints = tracker_cfg.get("max_keypoints", 20)
        
        # Re-ID feature gallery for matching reappearing objects
        self.enable_reid = reid_cfg.get("enabled", True)
        self.feature_gallery = {}  # {track_id: {"feature": vec, "age": frames_since_death}}
        self.gallery_max_age = reid_cfg.get("gallery_max_age", 300)  # Keep features for N frames
        self.reid_threshold = reid_cfg.get("threshold", 0.7)  # Cosine similarity threshold
        
        logger.debug(f"PlaneTracker initialized: iou_thresh={self.iou_threshold}, max_age={self.max_age}, min_hits={self.min_hits}, reid={self.enable_reid}")
    
    @staticmethod
    def get_mask_iou(mask1, mask2):
        intersect = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersect / union if union > 0 else 0
    
    def _match_to_gallery(self, feature_vec: np.ndarray) -> int | None:
        """
        Try to match a feature vector against the gallery of dead tracks.
        Returns the track_id if a match is found, None otherwise.
        """
        if not self.feature_gallery:
            return None
        
        best_match_id = None
        best_similarity = self.reid_threshold
        
        for track_id, entry in self.feature_gallery.items():
            # Cosine similarity (vectors are already L2 normalized)
            similarity = np.dot(feature_vec, entry["feature"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = track_id
        
        if best_match_id is not None:
            logger.info(f"Re-ID match: new detection matched to track {best_match_id} (sim={best_similarity:.3f})")
        
        return best_match_id
    
    def _age_gallery(self):
        """Remove old entries from the feature gallery."""
        expired = [tid for tid, entry in self.feature_gallery.items() 
                   if entry["age"] > self.gallery_max_age]
        for tid in expired:
            logger.debug(f"Gallery entry {tid} expired")
            del self.feature_gallery[tid]
    
    def spin(self, results, frame):
        h, w = frame.shape[:2]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # For optical flow
        
        detections = []
        res = results[0]
        if res.masks is not None:
            raw_masks = res.masks.data.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for i, (cls, conf) in enumerate(zip(clss, confs)):
                if res.names[int(cls)].lower() == 'airplane' and conf > 0.3:
                    m = cv2.resize(raw_masks[i], (w, h)) > 0.5
                    detections.append({'bbox': boxes[i], 'mask': m})
        track_ids = list(self.trackers.keys())
        for tid in track_ids:
            self.trackers[tid].predict()
        if detections and self.trackers:
            cost_matrix = np.zeros((len(detections), len(track_ids)))
            for i, det in enumerate(detections):
                for j, tid in enumerate(track_ids):
                    cost_matrix[i, j] = 1 - self.get_mask_iou(det['mask'], self.trackers[tid].mask)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1 - self.iou_threshold):
                    tid = track_ids[c]
                    self.trackers[tid].update(detections[r]['bbox'], detections[r]['mask'], gray_frame)
                    matched_indices.append(r)
            detections = [d for i, d in enumerate(detections) if i not in matched_indices]
        for det in detections:
            # Try Re-ID before creating new track
            matched_id = None
            if self.enable_reid:
                feature_vec = extract_feature_vector(frame, det['mask'], det['bbox'])
                matched_id = self._match_to_gallery(feature_vec)
            
            if matched_id is not None:
                # Resurrect old track with same ID
                entry = self.feature_gallery.pop(matched_id)
                self.trackers[matched_id] = Tracklet(matched_id, det['bbox'], det['mask'], max_keypoints=self.max_keypoints)
                self.trackers[matched_id]._prev_gray = gray_frame
                logger.info(f"Track {matched_id} resurrected via Re-ID")
            else:
                # Create new track
                self.trackers[self.next_id] = Tracklet(self.next_id, det['bbox'], det['mask'], max_keypoints=self.max_keypoints)
                self.trackers[self.next_id]._prev_gray = gray_frame
                logger.debug(f"New track {self.next_id} created at bbox {det['bbox'][:2].astype(int).tolist()}")
                self.next_id += 1
        # Age gallery entries
        if self.enable_reid:
            for entry in self.feature_gallery.values():
                entry["age"] += 1
            self._age_gallery()
        
        for tid in list(self.trackers.keys()):
            t = self.trackers[tid]
            if t.miss_count > self.max_age:
                # Save to gallery before deleting (for Re-ID)
                if self.enable_reid and t.hits >= self.min_hits:
                    feature_vec = extract_feature_vector(frame, t.mask, t.bbox)
                    self.feature_gallery[tid] = {
                        "feature": feature_vec,
                        "age": 0
                    }
                    logger.debug(f"Track {tid} added to Re-ID gallery")
                logger.debug(f"Track {tid} deleted (age={t.age}, hits={t.hits}, miss_count={t.miss_count})")
                del self.trackers[tid]
                continue
            draw_track(frame, t)