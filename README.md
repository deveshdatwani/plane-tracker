# Plane Tracker

**Approach:** Prioritized extensive visualization to enable manual validation in the absence of robust ground truths.

Aircraft tracking with hangar enter/exit event detection.

### Simple Plane Add
![Demo with GT overlay and metrics](outputs/demo_metrics.gif)

### Night Plane Overlap
![Night Plane Overlap Demo](outputs/night_overlap_demo.gif)

---

## Module Interaction & Data Flow

**Per-frame processing (inside `run_processing`):**

```
Frame N
   │
   ▼
┌────────────────┐
│   Detector     │──▶ YOLO inference → bboxes + masks + confidence
│ (processing.py)│
└───────┬────────┘
        │ raw detections
        ▼
┌───────────────────────────────────────────────────────────────┐
│              HangarControlManager.handle_frame()              │
│                                                               │
│   ┌──────────────────┐                                        │
│   │  PlaneTracker    │                                        │
│   │   .spin()        │──▶ Kalman predict → Hungarian match    │
│   │                  │    → Kalman update → keypoint tracking │
│   └────────┬─────────┘                                        │
│            │ self.trackers (dict of Tracklets)                │
│            ▼                                                  │
│   ┌──────────────────┐                                        │
│   │  HangarControl   │                                        │
│   │   .spin()        │──▶ IoU with tripwire → enter/exit      │
│   └────────┬─────────┘    events + track history              │
│            │                                                  │
│            ▼                                                  │
│   Accumulate to self.results["frames"][frame_id]              │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│              HangarControlManager.draw_debug()                │
│                                                               │
│   draw_processing_debug() → trajectory curves                 │
│   draw_debug_overlay()    → FPS, metrics, counts              │
│   draw_hangar_tripwire()  → boundary visualization            │
│   draw_ground_truth()     → GT overlay + P/R/HOTA             │
│   draw_track()            → bbox, keypoints, mask, labels     │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
Output frame (display / write video / JSON)
```

---

## Data Ownership

| Variable | Location | What it stores | Persistence |
|----------|----------|----------------|-------------|
| `detections` | `PlaneTracker.spin()` | Raw YOLO bboxes + masks | **None** — local var, discarded each frame |
| `self.trackers` | `PlaneTracker` | Dict of active `Tracklet` objects | Current state only, no history |
| `track.keypoints` | `Tracklet` | Nx2 array of tracked keypoints | Updated each frame via optical flow |
| `track.keypoint_ids` | `Tracklet` | Persistent IDs for each keypoint | Survives across frames for same track |
| `track._trajectory` | `drawing.py` | Last 30 centroids per track | Visualization only, not persisted to JSON |
| `self.track_history` | `HangarControl` | Last 20 bboxes per track (deque) | Rolling window, deleted when track dies |
| `self.results["frames"]` | `HangarControlManager` | Per-frame track output | Written to JSON (tracks, not raw detections) |

**Note:** Raw detections are not stored historically. Only tracked output is persisted.

---

## Usage

```bash
# Basic run with visualization
python run.py --video data/simple_plane_add2.mp4

# With ground truth overlay and metrics
python run.py --video data/simple_plane_add2.mp4 --annotations annotations/simple_plane_add2.json

# Save output video
python run.py --video data/simple_plane_add2.mp4 --save-video output.mp4

# Save JSON results with hangar events
python run.py --video data/simple_plane_add2.mp4 --output results.json

# Headless mode
python run.py --video data/simple_plane_add2.mp4 --no-display --output results.json
```

## Configuration

Edit `config.yaml` to adjust:

| Section | Key | Description |
|---------|-----|-------------|
| `detection` | model, confidence | YOLO model path and thresholds |
| `tracker` | iou_threshold, max_age | Tracking parameters |
| `hangar` | cooldown, flash_color | Event detection settings |
| `debug.level` | 0/1/2 | Overlay verbosity |
| `debug.show_ground_truth` | true/false | Show GT boxes + metrics |
| `debug.metrics_iou_threshold` | 0.0-1.0 | IOU threshold for P/R/HOTA |

## Metrics

When `show_ground_truth: true` with annotations loaded:
- **Detection** - Precision/Recall of raw YOLO detections vs GT
- **HOTA (ID-Agn)** - ID-agnostic tracking accuracy
- **HOTA (ID)** - ID-specific tracking accuracy (requires track ID match)

## Output Format

```json
{
  "frames": {
    "123": {
      "tracks": [{"track_id": 1, "bbox": [x1,y1,x2,y2], "class": "aircraft"}],
      "hangar_events": [{"track_id": 1, "event_type": "enter"}]
    }
  }
}
```
