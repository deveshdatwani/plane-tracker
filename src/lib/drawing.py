import cv2
import numpy as np
from typing import List, Dict

def _get_vis_config():
    """Get visualization config with defaults."""
    try:
        from src.config import get_config
        cfg = get_config()
        return cfg.get("visualization", {}), cfg.get("debug", {})
    except Exception:
        return {}, {}

def draw_hangar_tripwire(frame, hangar_control, frame_id=0):
    vis_cfg, _ = _get_vis_config()
    
    # Get hangar config for flash settings
    try:
        from src.config import get_config
        hangar_cfg = get_config().get("hangar", {})
    except Exception:
        hangar_cfg = {}
    
    boundary_color = tuple(vis_cfg.get("hangar_boundary_color", [60, 60, 60]))
    fill_color = (80, 80, 80)
    fill_alpha = 0.3
    
    # Check if we should flash (within cooldown period after an event)
    flash_on = hangar_cfg.get("flash_on_event", True)
    flash_color = tuple(hangar_cfg.get("flash_color", [0, 255, 255]))
    flash_rate = hangar_cfg.get("flash_rate", 4)
    cooldown = hangar_control.cooldown_frames
    
    is_flashing = False
    if flash_on and hasattr(hangar_control, 'last_event_frame'):
        frames_since_event = frame_id - hangar_control.last_event_frame
        if 0 < frames_since_event < cooldown:
            is_flashing = True
            # Toggle flash based on frame count
            if (frames_since_event // flash_rate) % 2 == 0:
                boundary_color = flash_color
                fill_color = flash_color
                fill_alpha = 0.2
    
    # Define hangar region coordinates
    if hangar_control.is_left_side:
        x1, y1 = 0, 0
        x2, y2 = hangar_control.boundary_x, hangar_control.frame_height
    else:
        x1, y1 = hangar_control.boundary_x, 0
        x2, y2 = hangar_control.frame_width, hangar_control.frame_height

    # Draw shaded overlay for hangar area
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    # Draw boundary rectangle
    line_thickness = 4 if is_flashing else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), boundary_color, line_thickness)

    # Add "Hangar Boundary" label (or event indicator when flashing)
    if is_flashing:
        label = "Event Detected"
        label_color = flash_color
    else:
        label = "Hangar Boundary"
        label_color = (220, 220, 220)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.55
    thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = x1 + 10
    text_y = y1 + text_size[1] + 10
    cv2.putText(frame, label, (text_x, text_y), font, font_scale, label_color, thickness, cv2.LINE_AA)


def draw_ground_truth(frame, gt_annotations, frame_id):
    """Draw ground truth annotations as dashed magenta boxes with 'GT' label.
    
    Args:
        frame: Video frame to draw on
        gt_annotations: Dict of frame_id -> list of annotations (already the frames dict)
        frame_id: Current frame ID
    """
    vis_cfg, debug_cfg = _get_vis_config()
    
    if not debug_cfg.get("show_ground_truth", False):
        return
    
    if gt_annotations is None:
        return
    
    # Frame IDs in JSON are strings
    frame_key = str(frame_id)
    
    if frame_key not in gt_annotations:
        return
    
    gt_color = tuple(vis_cfg.get("gt_color", [255, 0, 255]))
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = vis_cfg.get("font_scale", 0.45)
    thickness = vis_cfg.get("font_thickness", 1)
    
    for ann in gt_annotations[frame_key]:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Draw dashed rectangle (emulated with line segments)
        dash_length = 8
        gap_length = 6
        
        # Draw each side with dashes
        for x_start, y_start, x_end, y_end in [
            (x1, y1, x2, y1),  # Top
            (x2, y1, x2, y2),  # Right
            (x2, y2, x1, y2),  # Bottom
            (x1, y2, x1, y1),  # Left
        ]:
            dx = x_end - x_start
            dy = y_end - y_start
            length = np.sqrt(dx**2 + dy**2)
            if length == 0:
                continue
            dx, dy = dx / length, dy / length
            
            pos = 0
            drawing = True
            while pos < length:
                seg_len = dash_length if drawing else gap_length
                end_pos = min(pos + seg_len, length)
                if drawing:
                    px1 = int(x_start + dx * pos)
                    py1 = int(y_start + dy * pos)
                    px2 = int(x_start + dx * end_pos)
                    py2 = int(y_start + dy * end_pos)
                    cv2.line(frame, (px1, py1), (px2, py2), gt_color, 2)
                pos = end_pos
                drawing = not drawing
        
        # Draw 'GT' label above bbox
        track_id = ann.get("track_id", "")
        label = f"GT:{track_id}" if track_id else "GT"
        cv2.putText(frame, label, (x1, y1 - 8), font, font_scale, gt_color, thickness, cv2.LINE_AA)


def draw_corner_box(img, pt1, pt2, color, thickness, length=20):
    length = int(0.3 * (pt2[1] - pt1[1]))
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_track(frame, track):
    vis_cfg, debug_cfg = _get_vis_config()
    
    x1, y1, x2, y2 = track.bbox.astype(int)
    is_lost = track.miss_count > 0
    
    # Color coding from config
    color_active = tuple(vis_cfg.get("track_color_active", [200, 220, 100]))
    color_lost = tuple(vis_cfg.get("track_color_lost", [120, 120, 200]))
    tail_color = tuple(vis_cfg.get("tail_number_color", [0, 255, 255]))
    font_scale = vis_cfg.get("font_scale", 0.45)
    thickness = vis_cfg.get("font_thickness", 1)
    
    color = color_lost if is_lost else color_active
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Draw corner box
    draw_corner_box(frame, (x1, y1), (x2, y2), color, 1)
    
    # Always show track info on bounding box (independent of debug overlay level)
    line1 = f"ID:{track.id}  age:{track.age}"
    y_offset = y1 - 8
    cv2.putText(frame, line1, (x1, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    
    line2 = f"hits:{track.hits}  miss:{track.miss_count}"
    y_offset = y1 - 22
    cv2.putText(frame, line2, (x1, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Draw tail number if available (same style as track info, below bbox)
    tail_number = getattr(track, 'tail_number', None)
    if tail_number:
        tail_y = y2 + 15
        cv2.putText(frame, tail_number, (x1, tail_y), font, font_scale, tail_color, thickness, cv2.LINE_AA)
    
    # Draw status badge
    if is_lost:
        badge = "LOST"
        badge_color = (80, 80, 180)
    else:
        badge = "ACTIVE"
        badge_color = (80, 160, 80)
    cv2.putText(frame, badge, (x2 - 50, y1 - 8), font, font_scale, badge_color, thickness, cv2.LINE_AA)
    
    # Draw mask overlay for active tracks
    if not is_lost and track.mask is not None:
        overlay = frame.copy()
        overlay[track.mask] = color
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def _compute_iou(box1, box2):
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
    "det_tp": 0, "det_fp": 0, "det_fn": 0,  # Detection metrics
    "hota_agnostic_tp": 0, "hota_agnostic_fp": 0, "hota_agnostic_fn": 0, "hota_agnostic_iou_sum": 0,  # ID-agnostic
    "hota_id_tp": 0, "hota_id_fp": 0, "hota_id_fn": 0, "hota_id_iou_sum": 0,  # ID-specific
    "last_frame_id": -1
}


def _compute_all_metrics(detections, tracks, gt_annotations, frame_id, iou_threshold=0.5):
    """Compute cumulative detection and HOTA metrics.
    
    Returns dict with:
        - det_precision, det_recall
        - hota_agnostic (ID-agnostic HOTA)
        - hota_id (ID-specific HOTA)
    """
    global _cumulative_metrics
    
    # Reset if starting over
    if frame_id < _cumulative_metrics["last_frame_id"]:
        _cumulative_metrics = {
            "det_tp": 0, "det_fp": 0, "det_fn": 0,
            "hota_agnostic_tp": 0, "hota_agnostic_fp": 0, "hota_agnostic_fn": 0, "hota_agnostic_iou_sum": 0,
            "hota_id_tp": 0, "hota_id_fp": 0, "hota_id_fn": 0, "hota_id_iou_sum": 0,
            "last_frame_id": -1
        }
    
    # Only process each frame once
    if frame_id > _cumulative_metrics["last_frame_id"]:
        # Extract detection bboxes
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
        # tracks is a dict: {track_id: Tracklet}
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
                iou = _compute_iou(det_box, gt_box)
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
                iou = _compute_iou(track_box, gt_box)
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
                # Require track ID to match GT track ID
                if track_id != gt_tid:
                    continue
                iou = _compute_iou(track_box, gt_box)
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
    
    # Detection metrics
    det_tp, det_fp, det_fn = m["det_tp"], m["det_fp"], m["det_fn"]
    det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 1.0
    det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 1.0
    
    # ID-agnostic HOTA: sqrt(DetA * LocA)
    # DetA = TP / (TP + FP + FN), LocA = avg IoU of TPs
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


def draw_debug_overlay(frame: np.ndarray, frame_id: int, detections, tracks: List[Dict], pipeline_times: List[float], gt_annotations=None) -> np.ndarray:
    vis_cfg, debug_cfg = _get_vis_config()
    debug_level = debug_cfg.get("level", 2)
    
    if debug_level == 0:
        return frame
    
    h, w = frame.shape[:2]
    avg_ms = (sum(pipeline_times[-30:]) / min(len(pipeline_times), 30)) * 1000 if pipeline_times else 0
    fps_equiv = 1000 / avg_ms if avg_ms > 0 else 0
    
    # Count actual airplane detections from YOLO results
    det_count = 0
    try:
        res = detections[0]
        if res.boxes is not None:
            clss = res.boxes.cls.cpu().numpy()
            for cls in clss:
                if res.names[int(cls)].lower() == 'airplane':
                    det_count += 1
    except (IndexError, AttributeError):
        det_count = 0
    
    # Build lines based on debug config
    lines = []
    if debug_cfg.get("show_frame_id", True):
        lines.append(f"Frame: {frame_id}")
    if debug_cfg.get("show_detections", True):
        lines.append(f"Detections: {det_count}")
    if debug_cfg.get("show_tracks", True):
        lines.append(f"Tracks: {len(tracks)}")
    if debug_cfg.get("show_latency", True):
        lines.append(f"Latency: {avg_ms:.1f}ms")
    if debug_cfg.get("show_fps", True):
        lines.append(f"FPS: {fps_equiv:.1f}")
    
    # Show cumulative metrics when ground truth overlay is enabled
    if debug_cfg.get("show_ground_truth", False) and gt_annotations:
        iou_thresh = debug_cfg.get("metrics_iou_threshold", 0.5)
        metrics = _compute_all_metrics(detections, tracks, gt_annotations, frame_id, iou_threshold=iou_thresh)
        
        # Detection Metrics
        lines.append(f"---")
        lines.append(f"Detection")
        lines.append(f"P:{metrics['det_precision']*100:.0f}% R:{metrics['det_recall']*100:.0f}%")
        
        # ID-Agnostic HOTA
        lines.append(f"---")
        lines.append(f"HOTA (ID-Agn)")
        lines.append(f"{metrics['hota_agnostic']*100:.1f}%")
        
        # ID-Specific HOTA
        lines.append(f"---")
        lines.append(f"HOTA (ID)")
        lines.append(f"{metrics['hota_id']*100:.1f}%")
    
    if not lines:
        return frame
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = vis_cfg.get("font_scale", 0.45)
    thickness = vis_cfg.get("font_thickness", 1)
    
    # Draw tinted background for text readability
    box_x1 = w - 150
    box_y1 = 5
    box_x2 = w - 5
    box_y2 = 10 + len(lines) * 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    y = 20
    for line in lines:
        cv2.putText(frame, line, (w - 140, y), font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        y += 16
    return frame

def draw_fyveby_gt(frame, annotations) -> None:
    if not annotations: return 
    vis_cfg, _ = _get_vis_config()
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = vis_cfg.get("font_scale", 0.45)
    thickness = vis_cfg.get("font_thickness", 1)
    for ann in annotations:
        tid = ann["track_id"]
        x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"GT ID:{tid}  {ann.get('class', 'aircraft')}"
        cv2.putText(frame, label, (x1, y1 - 8), font, font_scale, color, thickness, cv2.LINE_AA)

TRACK_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (255, 180, 50),  (50, 180, 255),  (180, 255, 50),
]

def _color_for_id(track_id: int):
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# Module-level storage for historic bounding box crops (for processing debug)
_historic_bbox_crops = {}  # {track_id: [(x1, y1, crop_image), ...]}


def draw_processing_debug(frame, detections, trackers):
    """
    Draw all raw detections and trajectory curves for processing debug mode.
    Shows only bounding box regions on a black background, with all historic boxes persisted.
    """
    global _historic_bbox_crops
    
    _, debug_cfg = _get_vis_config()
    if not debug_cfg.get("processing_debug", False):
        return
    
    h, w = frame.shape[:2]
    
    # Store current bounding box crops before we modify the frame
    for tid, track in trackers.items():
        if hasattr(track, 'bbox') and track.miss_count == 0:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2].copy()
                
                if tid not in _historic_bbox_crops:
                    _historic_bbox_crops[tid] = []
                
                # Store with position - keep all history
                _historic_bbox_crops[tid].append((x1, y1, crop))
    
    # Clear dead tracks from history
    active_ids = set(trackers.keys())
    for tid in list(_historic_bbox_crops.keys()):
        if tid not in active_ids:
            # Keep for a bit after track dies, then clear
            pass  # Actually keep them for visualization
    
    # Create black frame
    frame[:] = 0
    
    # Draw all historic bounding box regions
    for tid, crops in _historic_bbox_crops.items():
        for x1, y1, crop in crops:
            ch, cw = crop.shape[:2]
            # Clamp to frame bounds
            x2, y2 = min(x1 + cw, w), min(y1 + ch, h)
            cw_actual = x2 - x1
            ch_actual = y2 - y1
            
            if cw_actual > 0 and ch_actual > 0:
                frame[y1:y2, x1:x2] = crop[:ch_actual, :cw_actual]
    
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    thickness = 1
    
    # Draw raw detections from YOLO (dashed rectangles)
    det_idx = 0
    try:
        res = detections[0]
        if res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                class_name = res.names[int(cls)]
                x1, y1, x2, y2 = [int(v) for v in box]
                
                # Use dashed rectangle for raw detections
                color = (180, 180, 180)  # Light gray for raw detections
                
                # Draw dashed rectangle
                dash_length = 8
                for j in range(x1, x2, dash_length * 2):
                    cv2.line(frame, (j, y1), (min(j + dash_length, x2), y1), color, 1)
                    cv2.line(frame, (j, y2), (min(j + dash_length, x2), y2), color, 1)
                for j in range(y1, y2, dash_length * 2):
                    cv2.line(frame, (x1, j), (x1, min(j + dash_length, y2)), color, 1)
                    cv2.line(frame, (x2, j), (x2, min(j + dash_length, y2)), color, 1)
                
                # Label with detection index and confidence
                label = f"det{det_idx} {class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y2 + 12), font, font_scale, color, thickness, cv2.LINE_AA)
                det_idx += 1
    except (IndexError, AttributeError):
        pass
    
    # Draw trajectory curves for each track
    for tid, track in trackers.items():
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        
        # Collect trajectory points from track's Kalman state history
        # We'll use the bbox center points
        if hasattr(track, 'bbox'):
            x1, y1, x2, y2 = track.bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_diag = np.sqrt(bbox_w**2 + bbox_h**2)
            
            # Store trajectory with bbox size context: (cx, cy, bbox_diagonal)
            if not hasattr(track, '_trajectory'):
                track._trajectory = []
            
            # Movement threshold is proportional to bbox size (larger object = more tolerance)
            movement_thresh = bbox_diag * 0.05  # 5% of diagonal
            
            # Check if this point should be added (significant movement relative to bbox size)
            should_add = True
            if len(track._trajectory) > 0:
                last_pt = track._trajectory[-1]
                dist = np.sqrt((cx - last_pt[0])**2 + (cy - last_pt[1])**2)
                avg_diag = (bbox_diag + last_pt[2]) / 2
                # For large objects, filter minor jitter
                if dist < avg_diag * 0.03:  # Less than 3% of avg bbox = noise
                    should_add = False
            
            if should_add:
                track._trajectory.append((cx, cy, bbox_diag))
            
            # Draw trajectory curve - use only last 20 points for smoothing
            # This handles direction changes better (e.g., airplane stopping and reversing)
            if len(track._trajectory) >= 2:
                # Use last 20 points for curve fitting
                recent_trajectory = track._trajectory[-20:]
                raw_points = np.array([(p[0], p[1]) for p in recent_trajectory], dtype=np.float64)
                bbox_sizes = np.array([p[2] for p in recent_trajectory], dtype=np.float64)
                n_pts = len(raw_points)
                
                # Fit polynomial curve with weighted least squares
                # Weight by inverse bbox size variance - more stable detections have higher weight
                if n_pts >= 4:
                    try:
                        # Parameterize by cumulative distance
                        t = np.zeros(n_pts)
                        for i in range(1, n_pts):
                            t[i] = t[i-1] + np.sqrt((raw_points[i, 0] - raw_points[i-1, 0])**2 + 
                                                     (raw_points[i, 1] - raw_points[i-1, 1])**2)
                        
                        if t[-1] > 0:
                            t = t / t[-1]  # Normalize to [0, 1]
                        else:
                            t = np.linspace(0, 1, n_pts)
                        
                        # Weights: larger bbox = more stable = higher weight
                        weights = bbox_sizes / np.max(bbox_sizes)
                        weights = np.clip(weights, 0.3, 1.0)  # Don't let small detections be ignored
                        
                        # Polynomial degree adapts (lower for fewer points)
                        deg = min(3, max(2, n_pts // 8))
                        
                        # Weighted least squares fit for x(t) and y(t)
                        coeffs_x = np.polyfit(t, raw_points[:, 0], deg, w=weights)
                        coeffs_y = np.polyfit(t, raw_points[:, 1], deg, w=weights)
                        
                        # Calculate residuals to identify outliers
                        fitted_x = np.polyval(coeffs_x, t)
                        fitted_y = np.polyval(coeffs_y, t)
                        residuals = np.sqrt((raw_points[:, 0] - fitted_x)**2 + 
                                           (raw_points[:, 1] - fitted_y)**2)
                        
                        # Outlier threshold scales with bbox size
                        outlier_thresh = bbox_sizes * 0.15  # 15% of bbox diagonal
                        inlier_mask = residuals < outlier_thresh
                        
                        # Refit with inliers only for cleaner curve
                        if np.sum(inlier_mask) >= 4:
                            t_inlier = t[inlier_mask]
                            pts_inlier = raw_points[inlier_mask]
                            w_inlier = weights[inlier_mask]
                            
                            coeffs_x = np.polyfit(t_inlier, pts_inlier[:, 0], deg, w=w_inlier)
                            coeffs_y = np.polyfit(t_inlier, pts_inlier[:, 1], deg, w=w_inlier)
                        
                        # Generate smooth curve
                        t_smooth = np.linspace(0, 1, max(n_pts * 2, 50))
                        smooth_x = np.polyval(coeffs_x, t_smooth)
                        smooth_y = np.polyval(coeffs_y, t_smooth)
                        
                        smooth_points = np.column_stack([smooth_x, smooth_y]).astype(np.int32)
                        
                        # Draw the fitted curve with gradient
                        for i in range(len(smooth_points) - 1):
                            alpha = i / len(smooth_points)
                            pt_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
                            cv2.line(frame, tuple(smooth_points[i]), tuple(smooth_points[i + 1]), pt_color, 2, cv2.LINE_AA)
                        
                        # Mark outlier detections with red X (only for recent points)
                        outlier_indices = np.where(~inlier_mask)[0]
                        for idx in outlier_indices:
                            ox, oy = int(raw_points[idx, 0]), int(raw_points[idx, 1])
                            cv2.drawMarker(frame, (ox, oy), (0, 0, 200), cv2.MARKER_TILTED_CROSS, 8, 2)
                        
                        # Prune outliers from recent history only
                        # Map back to full trajectory indices
                        if len(outlier_indices) > 0 and len(track._trajectory) > 20:
                            offset = len(track._trajectory) - 20
                            # Remove outliers from the recent window
                            full_inlier_indices = [offset + i for i in range(n_pts) if inlier_mask[i]]
                            old_part = track._trajectory[:offset]
                            new_part = [track._trajectory[i] for i in full_inlier_indices]
                            track._trajectory = old_part + new_part
                            
                    except Exception:
                        # Fallback to simple polylines
                        cv2.polylines(frame, [raw_points.astype(np.int32)], False, color, 2, cv2.LINE_AA)
                else:
                    # Simple line for few points
                    cv2.polylines(frame, [raw_points.astype(np.int32)], False, color, 2, cv2.LINE_AA)
                
                # Draw current position marker
                cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)