# Plane Tracker

**Approach:** Prioritized extensive visualization to enable manual validation in the absence of robust ground truths.

Aircraft tracking with hangar enter/exit event detection.

### Simple Plane Add
![Demo with GT overlay and metrics](outputs/demo_metrics.gif)

### Boston Airport
![Boston Airport Demo](outputs/boston_demo.gif)

### Night Plane Overlap
![Night Plane Overlap Demo](outputs/night_overlap_demo.gif)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              run.py (Entry Point)                           │
│   - Parses CLI args (--video, --annotations, --output, --save-video)        │
│   - Loads config.yaml via src/config.py                                     │
│   - Initializes Detector, HangarControlManager                              │
│   - Calls run_processing() main loop                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌──────────────────┐    ┌─────────────────────────┐    ┌──────────────────┐
│   src/config.py  │    │  src/processing.py      │    │  src/lib/utils.py│
│                  │    │                         │    │                  │
│ - load_config()  │    │  Detector               │    │ - load_annotations│
│ - get_config()   │    │   - YOLOv8 inference    │    │ - get_masked_frame│
│ - DEFAULTS dict  │    │   - bboxes + masks      │    └──────────────────┘
└──────────────────┘    │                         │
                        │  run_processing()       │
                        │   - Main frame loop     │
                        └────────────┬────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          src/hangar.py                                      │
│                                                                             │
│  HangarControlManager (orchestrator)                                        │
│   │                                                                         │
│   ├──▶ PlaneTracker (src/lib/tracker.py)                                    │
│   │      - Kalman filter + Hungarian matching                               │
│   │      - Tracklet: bbox, mask, keypoints, state                           │
│   │      - Optical flow keypoint tracking                                   │
│   │                                                                         │
│   ├──▶ HangarControl                                                        │
│   │      - Tripwire region + IoU gating                                     │
│   │      - Enter/exit event detection                                       │
│   │      - Track history (20-bbox rolling window)                           │
│   │                                                                         │
│   └──▶ Results accumulation (frames → JSON)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        src/lib/drawing.py                                   │
│                                                                             │
│  Visualization (called by HangarControlManager.draw_debug)                  │
│   - draw_track(): bbox, keypoints, mask overlay, labels                     │
│   - draw_hangar_tripwire(): boundary + flash on events                      │
│   - draw_debug_overlay(): FPS, latency, metrics                             │
│   - draw_ground_truth(): GT overlay + P/R/HOTA computation                  │
│   - draw_processing_debug(): trajectory curves (Catmull-Rom splines)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
