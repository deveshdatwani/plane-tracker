"""
Utility functions for the SkyView tracking pipeline.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def save_annotations(annotations: Dict, path: str):
    """
    Save annotations dict back to JSON.
    Inverse of load_annotations.
    """
    frames = {str(k): v for k, v in annotations.items()}
    with open(path, "w") as f:
        json.dump({"frames": frames}, f, indent=2)


# --- Visualization --


def draw_tracks(
    frame: np.ndarray,
    tracks: List[Dict],
    hangar_events: Optional[List] = None
) -> np.ndarray:
    """
    Draw bounding boxes and track IDs onto a frame with safety clipping.
    """
    h, w = frame.shape[:2]

    logger.debug(f"Drawing {len(tracks)} tracks on frame")

    for track in tracks:
        track_id = track["track_id"]
        
        # 1. Safety Unpack & Clip
        # We ensure coordinates are within the actual image resolution
        raw_coords = track["bbox"]
        x1 = max(0, min(int(raw_coords[0]), w))
        y1 = max(0, min(int(raw_coords[1]), h))
        x2 = max(0, min(int(raw_coords[2]), w))
        y2 = max(0, min(int(raw_coords[3]), h))

        # 2. Skip drawing if the box has no area (malformed Kalman prediction)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        color = _color_for_id(track_id)
        state = track.get("state", "confirmed")

        # Visual weight based on state
        thickness = 2 if state == "confirmed" else 1
        line_type = cv2.LINE_AA

        # Draw Rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)
        
        # Label with ID
        label = f"ID:{track_id}"
        # Draw background for text to make it readable
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, line_type)

    if hangar_events:
        for i, event in enumerate(hangar_events):
            if hasattr(event, 'to_dict'):
                event = event.to_dict()
            
            # Draw event notification in the top corner (staggered if multiple)
            label = f"EVENT: ID:{event['track_id']} {event['event_type'].upper()}"
            y_offset = 50 + (i * 30)
            cv2.putText(frame, label, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


# --- Rolling performance timer ---

class FrameTimer:
    """Simple rolling average timer for profiling pipeline stages."""

    def __init__(self, window: int = 30):
        self.window = window
        self._times: Dict[str, List[float]] = {}
        self._starts: Dict[str, float] = {}

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str):
        if name not in self._starts:
            return
        elapsed = time.perf_counter() - self._starts.pop(name)
        if name not in self._times:
            self._times[name] = []
        self._times[name].append(elapsed * 1000)
        if len(self._times[name]) > self.window:
            self._times[name].pop(0)

    def avg_ms(self, name: str) -> float:
        times = self._times.get(name, [])
        return sum(times) / len(times) if times else 0.0

    def summary(self) -> str:
        return " | ".join(f"{k}: {self.avg_ms(k):.1f}ms" for k in self._times)

def load_annotations(path: str) -> Dict[int, List[Dict]]:
    with open(path) as f:
        fyve_by_json = json.load(f)["frames"]
    return fyve_by_json

def get_masked_frame(frame, height, width, hangar_monitor):
    y_pad = int(height * 0.05)
    x_pad = int(width * 0.05)
    masked_frame = frame.copy()
    masked_frame[:y_pad, :] = 0
    masked_frame[-y_pad:, :] = 0
    if hangar_monitor.is_left_side:
        masked_frame[:, :x_pad] = 0
    else:
        masked_frame[:, -x_pad:] = 0
    return masked_frame
