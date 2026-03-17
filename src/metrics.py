"""
Part 4 — Evaluation Metrics
============================
Choose and implement three metrics to evaluate your full tracking pipeline.

Your task:
    - Choose three metrics appropriate for multi-object tracking
    - Implement or integrate them below
    - Compute them against the provided ground truth annotations

You may use open-source libraries (e.g. TrackEval, motmetrics). If you do,
make sure you can explain exactly what each metric computes.

Questions to answer in your write-up:
    1. Which three metrics did you choose and why?
    2. Why are these metrics appropriate for this specific domain?
    3. What does each metric tell you that the others don't?
    4. How does your pipeline perform? Walk us through the numbers.
    5. Are there cases where a high score on one metric masks poor performance
       on another?
"""

from typing import List, Dict
import numpy as np


class TrackingMetrics:
    """
    Accumulates per-frame tracking results and computes summary metrics.

    Must implement at least 3 metrics.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def update(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        frame_id: int
    ):
        """
        Accumulate results for one frame.

        Args:
            predictions:  List of track dicts from Tracker.update()
                          Each has: track_id, bbox [x1,y1,x2,y2], state
            ground_truth: List of GT annotation dicts for this frame
                          Each has: track_id, bbox [x1,y1,x2,y2]
            frame_id:     Current frame index
        """
        raise NotImplementedError

    def compute(self) -> Dict:
        """
        Compute and return all metrics over the accumulated frames.

        Returns:
            Dict mapping metric name -> value. Must include at least 3 metrics.
        """
        raise NotImplementedError

    def summary_string(self) -> str:
        metrics = self.compute()
        lines = ["=== Tracking Metrics ==="]
        for k, v in metrics.items():
            lines.append(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
        return "\n".join(lines)
