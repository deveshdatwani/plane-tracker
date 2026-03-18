from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import time
import logging
import cv2
# Delay importing heavy third-party libs until needed
YOLO = None
torch = None


@dataclass
class AnchorPoint:
    """Represents a tracked anchor point for frame stabilization."""
    x: int
    y: int
    template: np.ndarray
    confidence: float = 1.0
    original_x: int = 0
    original_y: int = 0
    is_inlier: bool = True
    
    def __post_init__(self):
        self.original_x = self.x
        self.original_y = self.y


class FrameStabilizer:
    """
    Anti-wobble frame stabilizer using template-matched anchor points.
    
    Creates anchor points on the first frame based on good features to track,
    then template matches them every frame to detect camera movement.
    The movement is corrected using a translation vector.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.num_anchors = config.get("num_anchors", 8)
        self.anchor_box_size = config.get("anchor_box_size", 32)
        self.search_margin = config.get("search_margin", 16)
        self.min_confidence = config.get("min_confidence", 0.7)
        self.reinit_threshold = config.get("reinit_threshold", 0.5)
        
        corner_cfg = config.get("corner_detection", {})
        self.corner_quality = corner_cfg.get("quality_level", 0.01)
        self.corner_min_distance = corner_cfg.get("min_distance", 50)
        self.corner_block_size = corner_cfg.get("block_size", 7)
        
        vis_cfg = config.get("visualization", {})
        self.vis_enabled = vis_cfg.get("enabled", True)
        self.draw_anchors = vis_cfg.get("draw_anchors", True)
        self.draw_translation = vis_cfg.get("draw_translation", True)
        self.draw_original_box = vis_cfg.get("draw_original_box", True)
        self.anchor_color = tuple(vis_cfg.get("anchor_color", [0, 255, 255]))
        self.original_color = tuple(vis_cfg.get("original_color", [0, 0, 255]))
        self.corrected_color = tuple(vis_cfg.get("corrected_color", [0, 255, 0]))
        self.line_thickness = vis_cfg.get("line_thickness", 2)
        
        self.anchors: List[AnchorPoint] = []
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.cumulative_translation_x = 0.0
        self.cumulative_translation_y = 0.0
        self.initialized = False
        self.logger = logging.getLogger(__name__)
    
    def _detect_corners(self, gray: np.ndarray) -> np.ndarray:
        """Detect good features to track (corners) in the frame."""
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.num_anchors * 3,  # Detect more than needed to select best
            qualityLevel=self.corner_quality,
            minDistance=self.corner_min_distance,
            blockSize=self.corner_block_size
        )
        
        if corners is None:
            return np.array([])
        
        return corners.reshape(-1, 2)
    
    def _create_anchor_template(self, gray: np.ndarray, x: int, y: int) -> Optional[np.ndarray]:
        """Extract template around anchor point."""
        half_size = self.anchor_box_size // 2
        h, w = gray.shape[:2]
        
        # Ensure within bounds
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x + half_size)
        y2 = min(h, y + half_size)
        
        if x2 - x1 < half_size or y2 - y1 < half_size:
            return None
            
        return gray[y1:y2, x1:x2].copy()
    
    def initialize(self, frame: np.ndarray):
        """Initialize anchor points from the first frame."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        corners = self._detect_corners(gray)
        
        if len(corners) == 0:
            self.logger.warning("No corners detected for stabilization")
            return
        
        # Sort by response/quality and select top N
        # Re-detect with quality measurement
        corner_response = []
        for cx, cy in corners:
            x, y = int(cx), int(cy)
            # Use Harris corner response as confidence
            response = cv2.cornerHarris(gray, 2, 3, 0.04)
            if 0 <= y < response.shape[0] and 0 <= x < response.shape[1]:
                corner_response.append((x, y, response[y, x]))
        
        # Sort by response (strongest first), randomly shuffle equal values
        corner_response.sort(key=lambda c: (-c[2], np.random.random()))
        
        self.anchors = []
        for x, y, response in corner_response[:self.num_anchors]:
            template = self._create_anchor_template(gray, x, y)
            if template is not None:
                anchor = AnchorPoint(x=x, y=y, template=template, confidence=1.0)
                self.anchors.append(anchor)
        
        self.initialized = len(self.anchors) > 0
        self.cumulative_translation_x = 0.0
        self.cumulative_translation_y = 0.0
        self.logger.info(f"Stabilizer initialized with {len(self.anchors)} anchor points")
    
    def _match_anchor(self, gray: np.ndarray, anchor: AnchorPoint) -> Tuple[int, int, float]:
        """Template match anchor in current frame, return new position and confidence."""
        h, w = gray.shape[:2]
        half_size = self.anchor_box_size // 2
        margin = self.search_margin
        
        # Define search region
        search_x1 = max(0, anchor.x - half_size - margin)
        search_y1 = max(0, anchor.y - half_size - margin)
        search_x2 = min(w, anchor.x + half_size + margin)
        search_y2 = min(h, anchor.y + half_size + margin)
        
        search_region = gray[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.shape[0] < anchor.template.shape[0] or \
           search_region.shape[1] < anchor.template.shape[1]:
            return anchor.x, anchor.y, 0.0
        
        # Template matching
        result = cv2.matchTemplate(search_region, anchor.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Convert to frame coordinates
        new_x = search_x1 + max_loc[0] + half_size
        new_y = search_y1 + max_loc[1] + half_size
        
        return new_x, new_y, max_val
    
    def stabilize(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Stabilize frame by detecting and correcting camera wobble.
        
        Returns:
            Tuple of (stabilized_frame, translation_x, translation_y)
        """
        if not self.initialized:
            self.initialize(frame)
            return frame, 0.0, 0.0
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Match ALL anchors to get translation measurements
        measurements = []  # List of (dx, dy, confidence, anchor)
        
        for anchor in self.anchors:
            new_x, new_y, confidence = self._match_anchor_from_original(gray, anchor)
            dx = new_x - anchor.original_x
            dy = new_y - anchor.original_y
            measurements.append((dx, dy, confidence, anchor, new_x, new_y))
        
        # Use RANSAC to find robust global translation
        self.translation_x, self.translation_y, inlier_mask = self._ransac_translation(measurements)
        
        # Update all anchors with the global translation (entire image moves together)
        for i, (dx, dy, confidence, anchor, new_x, new_y) in enumerate(measurements):
            # Set current position based on global translation for visualization
            anchor.x = int(anchor.original_x + self.translation_x)
            anchor.y = int(anchor.original_y + self.translation_y)
            anchor.confidence = confidence
            anchor.is_inlier = inlier_mask[i] if inlier_mask is not None else True
        
        # Check if we need to reinitialize (too few good matches)
        good_matches = sum(1 for m in measurements if m[2] >= self.min_confidence)
        if good_matches < len(self.anchors) * self.reinit_threshold:
            self.logger.info("Anchor confidence dropped, reinitializing stabilizer")
            self.initialize(frame)
            return frame, 0.0, 0.0
        
        # Store for visualization
        self.cumulative_translation_x = self.translation_x
        self.cumulative_translation_y = self.translation_y
        
        # Apply correction (translate in opposite direction)
        if abs(self.translation_x) > 0.1 or abs(self.translation_y) > 0.1:
            M = np.float32([
                [1, 0, -self.translation_x],
                [0, 1, -self.translation_y]
            ])
            stabilized = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            return stabilized, self.translation_x, self.translation_y
        
        return frame, 0.0, 0.0
    
    def _ransac_translation(self, measurements: List, threshold: float = 3.0, iterations: int = 100) -> Tuple[float, float, List[bool]]:
        """
        RANSAC estimation of global translation from anchor measurements.
        
        Args:
            measurements: List of (dx, dy, confidence, anchor, new_x, new_y)
            threshold: Inlier distance threshold in pixels
            iterations: Number of RANSAC iterations
        
        Returns:
            (translation_x, translation_y, inlier_mask)
        """
        # Filter by confidence first
        valid = [(i, m) for i, m in enumerate(measurements) if m[2] >= self.min_confidence]
        
        if len(valid) == 0:
            return 0.0, 0.0, [False] * len(measurements)
        
        if len(valid) == 1:
            idx, m = valid[0]
            mask = [False] * len(measurements)
            mask[idx] = True
            return float(m[0]), float(m[1]), mask
        
        best_tx, best_ty = 0.0, 0.0
        best_inliers = []
        best_count = 0
        
        translations = [(m[0], m[1]) for _, m in valid]
        indices = [i for i, _ in valid]
        
        for _ in range(iterations):
            # Random sample: pick one translation as hypothesis
            sample_idx = np.random.randint(len(translations))
            tx, ty = translations[sample_idx]
            
            # Count inliers
            inliers = []
            for j, (dx, dy) in enumerate(translations):
                dist = np.sqrt((dx - tx) ** 2 + (dy - ty) ** 2)
                if dist < threshold:
                    inliers.append(j)
            
            if len(inliers) > best_count:
                best_count = len(inliers)
                # Compute mean of inliers
                best_tx = np.mean([translations[j][0] for j in inliers])
                best_ty = np.mean([translations[j][1] for j in inliers])
                best_inliers = inliers
        
        # Build full inlier mask
        inlier_mask = [False] * len(measurements)
        for j in best_inliers:
            inlier_mask[indices[j]] = True
        
        return best_tx, best_ty, inlier_mask
    
    def _match_anchor_from_original(self, gray: np.ndarray, anchor: AnchorPoint) -> Tuple[int, int, float]:
        """Template match anchor searching around ORIGINAL position to detect drift."""
        h, w = gray.shape[:2]
        half_size = self.anchor_box_size // 2
        margin = self.search_margin
        
        # Search around ORIGINAL position
        search_x1 = max(0, anchor.original_x - half_size - margin)
        search_y1 = max(0, anchor.original_y - half_size - margin)
        search_x2 = min(w, anchor.original_x + half_size + margin)
        search_y2 = min(h, anchor.original_y + half_size + margin)
        
        search_region = gray[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.shape[0] < anchor.template.shape[0] or \
           search_region.shape[1] < anchor.template.shape[1]:
            return anchor.original_x, anchor.original_y, 0.0
        
        # Template matching
        result = cv2.matchTemplate(search_region, anchor.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Convert to frame coordinates
        new_x = search_x1 + max_loc[0] + half_size
        new_y = search_y1 + max_loc[1] + half_size
        
        return new_x, new_y, max_val
    
    def draw_visualization(self, frame: np.ndarray) -> np.ndarray:
        """Draw anchor points and translation visualization on frame."""
        if not self.vis_enabled or not self.initialized:
            return frame
        
        vis_frame = frame.copy()
        half_size = self.anchor_box_size // 2
        
        # Scale factor for visualizing small translations
        translation_scale = 10.0
        
        # Global translation (same for all anchors - entire image moves together)
        global_dx = self.translation_x
        global_dy = self.translation_y
        
        for anchor in self.anchors:
            # Determine color based on inlier status
            if anchor.is_inlier:
                box_color = self.anchor_color
                line_color = self.corrected_color
            else:
                box_color = (0, 100, 100)  # Dim yellow for outliers
                line_color = (100, 100, 100)  # Gray for outliers
            
            if self.draw_original_box:
                # Draw original position (red) - thin line
                cv2.rectangle(
                    vis_frame,
                    (anchor.original_x - half_size, anchor.original_y - half_size),
                    (anchor.original_x + half_size, anchor.original_y + half_size),
                    self.original_color,
                    1
                )
            
            if self.draw_anchors:
                # Draw current position based on GLOBAL translation
                current_x = int(anchor.original_x + global_dx)
                current_y = int(anchor.original_y + global_dy)
                cv2.rectangle(
                    vis_frame,
                    (current_x - half_size, current_y - half_size),
                    (current_x + half_size, current_y + half_size),
                    box_color,
                    1
                )
                
                # Draw confidence
                cv2.putText(
                    vis_frame,
                    f"{anchor.confidence:.2f}",
                    (current_x - half_size, current_y - half_size - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    box_color,
                    1
                )
            
            if self.draw_translation:
                # Draw scaled GLOBAL translation from each anchor's original position
                scaled_end_x = int(anchor.original_x + global_dx * translation_scale)
                scaled_end_y = int(anchor.original_y + global_dy * translation_scale)
                
                cv2.arrowedLine(
                    vis_frame,
                    (anchor.original_x, anchor.original_y),
                    (scaled_end_x, scaled_end_y),
                    line_color,
                    1,
                    tipLength=0.2
                )
        
        # Draw cumulative translation info
        if self.draw_translation:
            info_text = f"Translation: ({self.cumulative_translation_x:.1f}, {self.cumulative_translation_y:.1f})"
            cv2.putText(
                vis_frame,
                info_text,
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.corrected_color,
                1
            )
        
        return vis_frame


@dataclass
class Detection:
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str = ""
    frame_id: int = -1

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class Detector:
    def __init__(self, model_path="yolov8s-seg.pt", conf_thresh=0.4, iou_thresh=0.45, verbose=False):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.verbose = verbose
        global YOLO, torch
        try:
            if YOLO is None:
                from ultralytics import YOLO as _YOLO
                YOLO = _YOLO
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("ultralytics package is required for detection. Install via `pip install ultralytics`") from e
        try:
            if torch is None:
                import torch as _torch
                torch = _torch
        except ModuleNotFoundError:
            torch = None

        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray):
        # Return the raw ultralytics results for downstream processing
        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=self.verbose)
        return results


def run_processing(cap, detector, hangar_manager, writer=None, gt_annotations=None, no_display=False, start_frame=None, end_frame=None, gif_frames=None, stabilizer=None):
    import cv2
    import time

    logger = logging.getLogger(__name__)
    pipeline_times = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle frame range
    actual_start = start_frame if start_frame is not None else 0
    actual_end = end_frame if end_frame is not None else total_frames - 1
    
    if actual_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        logger.info(f"Seeking to frame {actual_start}")
    
    logger.info(f"Starting processing loop (frames {actual_start}-{actual_end}, {actual_end - actual_start + 1} frames)")
    if stabilizer:
        logger.info("Frame stabilization enabled")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Stop if past end frame
        if frame_id > actual_end + 1:
            break
        
        t_start = time.perf_counter()

        # Apply frame stabilization (anti-wobble) before detection
        stabilized_frame = frame
        translation_x, translation_y = 0.0, 0.0
        if stabilizer:
            stabilized_frame, translation_x, translation_y = stabilizer.stabilize(frame)

        # get masked frame if hangar manager provides masking helper
        if hasattr(hangar_manager, "get_masked_frame"):
            masked_frame = hangar_manager.get_masked_frame(stabilized_frame)
        else:
            masked_frame = stabilized_frame

        detections = detector.detect(masked_frame)

        # hand detections to hangar_control manager which will handle tracking and result generation
        hangar_manager.handle_frame(detections, stabilized_frame, frame_id)

        t_end = time.perf_counter()
        pipeline_times.append(t_end - t_start)

        # optional display / writer handled by hangar_manager
        vis_frame = None
        if hasattr(hangar_manager, "draw_debug"):
            vis_frame = hangar_manager.draw_debug(stabilized_frame, frame_id, detections, pipeline_times, gt_annotations)
        
        # Draw stabilizer visualization on top
        if stabilizer and vis_frame is not None:
            vis_frame = stabilizer.draw_visualization(vis_frame)

        if writer and vis_frame is not None:
            writer.write(vis_frame)
        
        # Collect frames for GIF if requested
        if gif_frames is not None and vis_frame is not None:
            # Convert BGR to RGB for PIL
            gif_frames.append(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))

        if not no_display and vis_frame is not None:
            cv2.imshow("SkyView - Aircraft Tracker", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info(f"User quit at frame {frame_id}")
                break

    avg_time = np.mean(pipeline_times) if pipeline_times else 0
    logger.info(f"Processing complete: {len(pipeline_times)} frames, avg {avg_time*1000:.1f}ms/frame")
    # return pipeline timings
    return pipeline_times
