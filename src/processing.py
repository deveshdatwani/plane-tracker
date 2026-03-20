from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import time
import logging
# Delay importing heavy third-party libs until needed
YOLO = None
torch = None


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


def run_processing(cap, detector, hangar_manager, writer=None, gt_annotations=None, no_display=False):
    import cv2
    import time

    logger = logging.getLogger(__name__)
    pipeline_times = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Starting processing loop ({total_frames} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        t_start = time.perf_counter()

        # get masked frame if hangar manager provides masking helper
        if hasattr(hangar_manager, "get_masked_frame"):
            masked_frame = hangar_manager.get_masked_frame(frame)
        else:
            masked_frame = frame

        detections = detector.detect(masked_frame)

        # hand detections to hangar_control manager which will handle tracking and result generation
        hangar_manager.handle_frame(detections, frame, frame_id)

        t_end = time.perf_counter()
        pipeline_times.append(t_end - t_start)

        # optional display / writer handled by hangar_manager
        vis_frame = None
        if hasattr(hangar_manager, "draw_debug"):
            vis_frame = hangar_manager.draw_debug(frame, frame_id, detections, pipeline_times, gt_annotations)

        if writer and vis_frame is not None:
            writer.write(vis_frame)

        if not no_display and vis_frame is not None:
            cv2.imshow("SkyView - Aircraft Tracker", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info(f"User quit at frame {frame_id}")
                break

    avg_time = np.mean(pipeline_times) if pipeline_times else 0
    logger.info(f"Processing complete: {len(pipeline_times)} frames, avg {avg_time*1000:.1f}ms/frame")
    # return pipeline timings
    return pipeline_times
