import numpy as np
import cv2
import logging
from scipy.optimize import linear_sum_assignment
from src.lib.drawing import draw_track
from src.processing import TailNumberExtractor

logger = logging.getLogger(__name__)

class Tracklet:
    def __init__(self, track_id, bbox, mask):
        self.id = track_id
        self.miss_count = 0
        self.mask = mask
        self.age = 0
        self.hits = 0
        self.tail_number = None  # Extracted via OCR
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
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.miss_count += 1
        cx, cy, w, h = self.x[:4].flatten()
        self.bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        return self.bbox
    
    def update(self, new_bbox, new_mask):
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
    def __init__(self, iou_threshold=None, max_age=None, enable_ocr=None, ocr_interval=None, min_hits=None):
        from src.config import get_config
        cfg = get_config()
        tracker_cfg = cfg.get("tracker", {})
        ocr_cfg = cfg.get("ocr", {})
        
        self.trackers = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold if iou_threshold is not None else tracker_cfg.get("iou_threshold", 0.2)
        self.max_age = max_age if max_age is not None else tracker_cfg.get("max_age", 30)
        self.min_hits = min_hits if min_hits is not None else tracker_cfg.get("min_hits", 10)
        self.enable_ocr = enable_ocr if enable_ocr is not None else ocr_cfg.get("enabled", True)
        self._ocr_extractor = TailNumberExtractor() if self.enable_ocr else None
        self._ocr_interval = ocr_interval if ocr_interval is not None else ocr_cfg.get("interval", 10)
        logger.debug(f"PlaneTracker initialized: iou_thresh={self.iou_threshold}, max_age={self.max_age}, min_hits={self.min_hits}, ocr={self.enable_ocr}")
    
    @staticmethod
    def get_mask_iou(mask1, mask2):
        intersect = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersect / union if union > 0 else 0
    
    def spin(self, results, frame):
        h, w = frame.shape[:2]
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
                    self.trackers[tid].update(detections[r]['bbox'], detections[r]['mask'])
                    matched_indices.append(r)
            detections = [d for i, d in enumerate(detections) if i not in matched_indices]
        for det in detections:
            self.trackers[self.next_id] = Tracklet(self.next_id, det['bbox'], det['mask'])
            logger.debug(f"New track {self.next_id} created at bbox {det['bbox'][:2].astype(int).tolist()}")
            self.next_id += 1
        for tid in list(self.trackers.keys()):
            t = self.trackers[tid]
            if t.miss_count > self.max_age:
                logger.debug(f"Track {tid} deleted (age={t.age}, hits={t.hits}, miss_count={t.miss_count})")
                del self.trackers[tid]
                continue
            # Try OCR for tail number periodically if not yet found
            # Run more frequently when track is young, then slow down
            if self.enable_ocr and self._ocr_extractor and t.tail_number is None:
                interval = 3 if t.age < 30 else self._ocr_interval
                if t.age % interval == 0 and t.miss_count == 0:
                    logger.debug(f"Track {tid}: triggering OCR (age={t.age}, hits={t.hits})")
                    tail = self._ocr_extractor.extract(frame, t.bbox)
                    if tail:
                        t.tail_number = tail
                        logger.info(f"Track {tid} tail number extracted: {tail}")
            draw_track(frame, t)