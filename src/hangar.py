from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from pathlib import Path
import json
import logging

from src.lib.tracker import PlaneTracker
from src.lib.utils import get_masked_frame
from src.lib.drawing import draw_hangar_tripwire, draw_debug_overlay, draw_processing_debug, draw_ground_truth

logger = logging.getLogger(__name__)

@dataclass
class HangarEvent:
    track_id: int
    frame_id: int
    event_type: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

class HangarControl:
    def __init__(self, frame_width, frame_height, seq, cooldown_frames=None, iou_threshold=None):
        from src.config import get_config
        cfg = get_config()
        hangar_cfg = cfg.get("hangar", {})
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.seq = seq.split("/")[-1]
        self.cooldown_frames = cooldown_frames if cooldown_frames is not None else hangar_cfg.get("cooldown_frames", 30)
        self.iou_threshold = iou_threshold if iou_threshold is not None else hangar_cfg.get("iou_threshold", 0.01)
        self.last_event_frame = -self.cooldown_frames
        if self.seq not in ["no_planes.mp4", "night_plane_overlap.mp4"]:
            self.boundary_x = int(self.frame_width * 0.15)
            self.hangar_bbox = [0, 0, self.boundary_x, self.frame_height]
            self.is_left_side = True
        else:
            self.boundary_x = int(self.frame_width * 0.85)
            self.hangar_bbox = [self.boundary_x, 0, self.frame_width, self.frame_height]
            self.is_left_side = False
        self.track_history = {}
        self.track_status = {}
        self.event_logs = []
        logger.debug(f"HangarControl initialized: side={'left' if self.is_left_side else 'right'}, boundary_x={self.boundary_x}, cooldown={self.cooldown_frames}")
    def _get_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]) , min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
    def spin(self, tracks, frame_id):
        new_events = []
        current_ids = set(tracks.keys())
        for t_id, t_data in tracks.items():
            if t_data.miss_count > 0 or t_data.hits < 10:
                continue
            bbox = t_data.bbox
            if t_id not in self.track_history:
                self.track_history[t_id] = deque(maxlen=20)
            self.track_history[t_id].append({"frame": frame_id, "bbox": bbox})
            iou = self._get_iou(bbox, self.hangar_bbox)
            is_inside = iou > self.iou_threshold
            if t_id not in self.track_status:
                self.track_status[t_id] = is_inside
                continue
            was_inside = self.track_status[t_id]
            event_type = None
            if is_inside and not was_inside: event_type = "enter"
            elif not is_inside and was_inside: event_type = "exit"
            if event_type and (frame_id - self.last_event_frame >= self.cooldown_frames):
                event = HangarEvent(track_id=t_id, frame_id=frame_id, event_type=event_type, metadata={"trajectory": list(self.track_history[t_id]), "iou": iou})
                self.event_logs.append(event)
                new_events.append(event)
                self.last_event_frame = frame_id
                logger.info(f"Hangar event: track {t_id} {event_type} at frame {frame_id} (iou={iou:.3f})")
            self.track_status[t_id] = is_inside
        for tid in list(self.track_history.keys()):
            if tid not in current_ids:
                self.track_history.pop(tid, None)
                self.track_status.pop(tid, None)
        return new_events


class HangarControlManager:
    """Wrapper that combines `PlaneTracker` with `HangarControl` and produces per-frame JSON results."""
    def __init__(self, frame_width, frame_height, seq, output_path=None, fps=None):
        self.tracker = PlaneTracker()
        self.hangar = HangarControl(frame_width=frame_width, frame_height=frame_height, seq=seq)
        self.results = {"frames": {}}
        self.output_path = output_path
        self._frame_width = frame_width
        self._frame_height = frame_height
        logger.info(f"HangarControlManager initialized for {seq} ({frame_width}x{frame_height} @ {fps}fps)")

    def get_masked_frame(self, frame):
        return get_masked_frame(frame, self._frame_height, self._frame_width, self.hangar)

    def handle_frame(self, detections, frame, frame_id):
        # update tracker
        self.tracker.spin(detections, frame)

        # check hangar events
        new_events = self.hangar.spin(self.tracker.trackers, frame_id)

        # compose labels similar to previous run.py format
        labels = [{
            "track_id": idx, 
            "bbox": tx.bbox.tolist(), 
            "class": "aircraft"
        } for idx, tx in self.tracker.trackers.items()]
        
        self.results["frames"][str(frame_id)] = labels

        return labels, new_events

    def draw_debug(self, frame, frame_id, detections, pipeline_times, gt_annotations=None):
        draw_processing_debug(frame, detections, self.tracker.trackers)
        vis = draw_debug_overlay(frame, frame_id, detections, self.tracker.trackers, pipeline_times, gt_annotations)
        draw_hangar_tripwire(frame, self.hangar, frame_id)
        draw_ground_truth(frame, gt_annotations, frame_id)
        return vis

    def write_output(self, path=None):
        out_path = path or self.output_path
        if not out_path:
            logger.warning("No output path specified, skipping write")
            return
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Wrote tracking results to {out}")