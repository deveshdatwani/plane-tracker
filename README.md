# Plane Tracker

**Approach:** Prioritized extensive visualization to enable manual validation in the absence of robust ground truths.

Aircraft tracking with hangar enter/exit event detection.

### Simple Plane Add
![Demo with GT overlay and metrics](outputs/demo_metrics.gif)

### Boston Airport
![Boston Airport Demo](outputs/boston_demo.gif)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              run.py (Entry Point)                           │
│   - Parses CLI args (--video, --annotations, --output, --save-video)        │
│   - Loads config.yaml via src/config.py                                     │
│   - Initializes Detector, HangarControlManager                              │
│   - Runs main processing loop                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        src/processing.py                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Detector                                                                   │
│   - Wraps YOLOv8 (yolov8n-seg.pt) for airplane detection                    │
│   - Outputs: bboxes, confidence scores, segmentation masks                  │
│   - Filters by class='airplane' and confidence threshold                   │
│                                                                             │
│  TailNumberExtractor                                                        │
│   - OCR module using pytesseract                                            │
│   - Extracts N-numbers from bbox crops using multiple preprocessing:        │
│     grayscale, CLAHE, thresholding, scaling                                 │
│   - Regex pattern matching for US tail numbers (N + digits)                 │
│                                                                             │
│  run_processing()                                                           │
│   - Main frame loop: read → detect → track → hangar events → visualize     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          src/hangar.py                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  HangarControlManager                                                       │
│   - Orchestrates PlaneTracker + HangarControl                               │
│   - Manages per-frame JSON result accumulation                              │
│   - Provides get_masked_frame() for hangar boundary visualization           │
│                                                                             │
│  HangarControl                                                              │
│   - Defines hangar tripwire region (left or right side of frame)            │
│   - Computes IoU between track bboxes and hangar boundary                   │
│   - Emits "enter"/"exit" events with cooldown to prevent duplicates         │
│   - Tracks per-aircraft state history for event metadata                    │
│                                                                             │
│  HangarEvent (dataclass)                                                    │
│   - track_id, frame_id, event_type, confidence, metadata                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        src/lib/tracker.py                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  PlaneTracker                                                               │
│   - Multi-object tracker using Kalman filter + Hungarian matching           │
│   - Data association: mask IoU cost matrix → scipy.linear_sum_assignment    │
│   - Track lifecycle: create (new det) → update (matched) → delete (max_age) │
│   - Optionally runs OCR on tracked bboxes at configurable intervals         │
│                                                                             │
│  Tracklet                                                                   │
│   - Per-object state: Kalman filter (8D state: cx,cy,w,h + velocities)      │
│   - predict(): propagate state, increment miss_count                        │
│   - update(): Kalman correction with new measurement, reset miss_count      │
│   - Stores: bbox, mask, track_id, age, hits, tail_number                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        src/lib/drawing.py                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Visualization Functions:                                                   │
│   - draw_hangar_tripwire(): Renders hangar boundary with flash on events    │
│   - draw_track(): Draws bbox, ID label, tail number, segmentation mask      │
│   - draw_debug_overlay(): FPS, latency, detection/track counts, metrics     │
│   - draw_ground_truth(): GT boxes overlay with precision/recall/HOTA        │
│   - draw_processing_debug(): Raw YOLO detections + trajectory curves        │
│                                                                             │
│  Trajectory Smoothing:                                                      │
│   - _catmull_rom_spline(): Interpolating spline through control points      │
│   - _remove_self_crossings(): Eliminates loops/knots from trajectories      │
│   - Direction consistency filtering to reject U-turn outliers               │
│                                                                             │
│  Metrics Computation:                                                       │
│   - _compute_all_metrics(): Detection P/R, HOTA (ID-agnostic & ID-specific) │
│   - _compute_iou(): Box IoU for matching predictions to GT                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        src/lib/utils.py                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  - load_annotations(): Parse GT JSON files                                  │
│  - get_masked_frame(): Apply hangar boundary mask to frame                  │
│  - Utility functions for bbox manipulation                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          src/config.py                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  - load_config(): Load config.yaml with YAML parser                         │
│  - get_config(): Global config accessor                                     │
│  - DEFAULTS dict: Fallback values for all config sections                   │
│                                                                             │
│  Config Sections:                                                           │
│   - detection: model_path, confidence_threshold, iou_threshold              │
│   - tracker: iou_threshold, max_age, min_hits                               │
│   - hangar: cooldown_frames, iou_threshold, flash settings                  │
│   - ocr: enabled, interval                                                  │
│   - debug: level (0/1/2), show_ground_truth, processing_debug               │
│   - visualization: colors for tracks, hangar, GT overlays                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          src/metrics.py                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  TrackingMetrics (abstract interface)                                       │
│   - update(): Accumulate per-frame predictions vs GT                        │
│   - compute(): Return dict of metric values                                 │
│   - Implementation in draw_debug_overlay computes:                          │
│     • Detection Precision/Recall                                            │
│     • HOTA (ID-Agnostic): sqrt(DetA × LocA)                                  │
│     • HOTA (ID-Specific): Requires track ID match with GT                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow (Per Frame)

```
Frame N from video
       │
       ▼
┌──────────────┐
│   Detector   │ ──▶ YOLO inference → bboxes + masks + confidence
└──────────────┘
       │
       ▼
┌──────────────┐
│ PlaneTracker │ ──▶ Kalman predict → Hungarian matching → Kalman update
└──────────────┘     Creates/updates/deletes Tracklets
       │
       ▼
┌──────────────┐
│ HangarControl│ ──▶ IoU with hangar boundary → "enter"/"exit" events
└──────────────┘
       │
       ▼
┌──────────────┐
│  Visualizer  │ ──▶ Draw tracks, tripwire, debug info, GT overlay
└──────────────┘
       │
       ▼
Output frame (display / save video / JSON results)
```

---

## Module Interaction

```
┌────────────────┐     raw YOLO      ┌──────────────┐    Tracklets     ┌───────────────┐
│    Detector    │ ──────────────▶   │ PlaneTracker │ ──────────────▶  │ HangarControl │
│ (processing.py)│   detections      │ (tracker.py) │   self.trackers  │  (hangar.py)  │
└────────────────┘                   └──────────────┘                  └───────────────┘
                                            │                                 │
                                            │                                 │
                                     Creates/updates              Keeps 20-bbox rolling
                                     Tracklet objects             history per track for
                                     (Kalman state,               event metadata
                                      current bbox, mask)
```

---

## Data Ownership

| Variable | Location | What it stores | Persistence |
|----------|----------|----------------|-------------|
| `detections` | `PlaneTracker.spin()` | Raw YOLO bboxes + masks | **None** — local var, discarded each frame |
| `self.trackers` | `PlaneTracker` | Dict of active `Tracklet` objects | Current state only, no history |
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
