"""
Evaluation Metrics for multi-object tracking.

Three metrics:
    1. Detection Precision/Recall — raw YOLO detections vs GT
    2. HOTA (ID-Agnostic) — tracking accuracy ignoring identity
    3. HOTA (ID-Specific) — tracking accuracy requiring correct identity
"""

from typing import Dict, List
import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Cumulative metrics storage
_cumulative_metrics = {
    "det_tp": 0, "det_fp": 0, "det_fn": 0,
    "hota_agnostic_tp": 0, "hota_agnostic_fp": 0, "hota_agnostic_fn": 0, "hota_agnostic_iou_sum": 0,
    "hota_id_tp": 0, "hota_id_fp": 0, "hota_id_fn": 0, "hota_id_iou_sum": 0,
    "last_frame_id": -1
}


def reset_cumulative_metrics():
    """Reset all accumulated metric state."""
    global _cumulative_metrics
    _cumulative_metrics = {
        "det_tp": 0, "det_fp": 0, "det_fn": 0,
        "hota_agnostic_tp": 0, "hota_agnostic_fp": 0, "hota_agnostic_fn": 0, "hota_agnostic_iou_sum": 0,
        "hota_id_tp": 0, "hota_id_fp": 0, "hota_id_fn": 0, "hota_id_iou_sum": 0,
        "last_frame_id": -1
    }


def compute_all_metrics(detections, tracks, gt_annotations, frame_id, iou_threshold=0.5):
    """Compute cumulative detection and HOTA metrics.

    Args:
        detections: Raw YOLO results list (ultralytics format).
        tracks: Dict of {track_id: Tracklet} or list of Tracklet objects.
        gt_annotations: Dict of frame_id (str) -> list of annotation dicts.
        frame_id: Current frame index.
        iou_threshold: IoU threshold for matching.

    Returns dict with:
        - det_precision, det_recall, det_tp, det_fp, det_fn
        - hota_agnostic (ID-agnostic HOTA)
        - hota_id (ID-specific HOTA)
    """
    global _cumulative_metrics

    # Reset if starting over
    if frame_id < _cumulative_metrics["last_frame_id"]:
        reset_cumulative_metrics()

    # Only process each frame once
    if frame_id > _cumulative_metrics["last_frame_id"]:
        # Extract detection bboxes from raw YOLO results
        det_boxes = []
        try:
            res = detections[0]
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy()
                for i, cls in enumerate(clss):
                    if res.names[int(cls)].lower() == 'airplane':
                        det_boxes.append(boxes[i])
        except (IndexError, AttributeError):
            pass

        # Extract track bboxes with IDs
        track_data = []  # (bbox, track_id)
        track_items = tracks.values() if isinstance(tracks, dict) else tracks
        for t in track_items:
            track_data.append((t.bbox, t.id))

        # Get GT boxes with IDs for this frame
        gt_data = []  # (bbox, track_id)
        if gt_annotations:
            frame_key = str(frame_id)
            if frame_key in gt_annotations:
                for ann in gt_annotations[frame_key]:
                    bbox = ann.get("bbox", [])
                    tid = ann.get("track_id", -1)
                    if len(bbox) == 4:
                        gt_data.append((bbox, tid))

        gt_boxes = [g[0] for g in gt_data]

        # --- Detection metrics (raw detections vs GT) ---
        matched_gt_det = set()
        for det_box in det_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt_det:
                    continue
                iou = compute_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                matched_gt_det.add(best_gt_idx)

        _cumulative_metrics["det_tp"] += len(matched_gt_det)
        _cumulative_metrics["det_fp"] += len(det_boxes) - len(matched_gt_det)
        _cumulative_metrics["det_fn"] += len(gt_boxes) - len(matched_gt_det)

        # --- ID-agnostic HOTA (tracks vs GT, ignore IDs) ---
        matched_gt_agnostic = set()
        for track_box, _ in track_data:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, (gt_box, _) in enumerate(gt_data):
                if gt_idx in matched_gt_agnostic:
                    continue
                iou = compute_iou(track_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                matched_gt_agnostic.add(best_gt_idx)
                _cumulative_metrics["hota_agnostic_iou_sum"] += best_iou
                _cumulative_metrics["hota_agnostic_tp"] += 1

        _cumulative_metrics["hota_agnostic_fp"] += len(track_data) - len(matched_gt_agnostic)
        _cumulative_metrics["hota_agnostic_fn"] += len(gt_data) - len(matched_gt_agnostic)

        # --- ID-specific HOTA (tracks vs GT, require ID match) ---
        matched_gt_id = set()
        for track_box, track_id in track_data:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, (gt_box, gt_tid) in enumerate(gt_data):
                if gt_idx in matched_gt_id:
                    continue
                if track_id != gt_tid:
                    continue
                iou = compute_iou(track_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                matched_gt_id.add(best_gt_idx)
                _cumulative_metrics["hota_id_iou_sum"] += best_iou
                _cumulative_metrics["hota_id_tp"] += 1

        _cumulative_metrics["hota_id_fp"] += len(track_data) - len(matched_gt_id)
        _cumulative_metrics["hota_id_fn"] += len(gt_data) - len(matched_gt_id)

        _cumulative_metrics["last_frame_id"] = frame_id

    # Compute final metrics
    m = _cumulative_metrics

    det_tp, det_fp, det_fn = m["det_tp"], m["det_fp"], m["det_fn"]
    det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 1.0
    det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 1.0

    # ID-agnostic HOTA: sqrt(DetA * LocA)
    ag_tp, ag_fp, ag_fn = m["hota_agnostic_tp"], m["hota_agnostic_fp"], m["hota_agnostic_fn"]
    ag_deta = ag_tp / (ag_tp + ag_fp + ag_fn) if (ag_tp + ag_fp + ag_fn) > 0 else 1.0
    ag_loca = m["hota_agnostic_iou_sum"] / ag_tp if ag_tp > 0 else 1.0
    hota_agnostic = (ag_deta * ag_loca) ** 0.5

    # ID-specific HOTA
    id_tp, id_fp, id_fn = m["hota_id_tp"], m["hota_id_fp"], m["hota_id_fn"]
    id_deta = id_tp / (id_tp + id_fp + id_fn) if (id_tp + id_fp + id_fn) > 0 else 1.0
    id_loca = m["hota_id_iou_sum"] / id_tp if id_tp > 0 else 1.0
    hota_id = (id_deta * id_loca) ** 0.5

    return {
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_tp": det_tp, "det_fp": det_fp, "det_fn": det_fn,
        "hota_agnostic": hota_agnostic,
        "hota_id": hota_id,
    }
