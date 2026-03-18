# Take-Home Write-Up

**Name:** Devesh Datwani  
**Date:** 03/16/2026  
**Time spent:** 4 hours

---

## Part 1 — Detection

1. Why did you choose this algorithm for this domain specifically?
```
a. I chose YOLOv8n to prioritize tracker robustness while reducing reliance on heavy models for edge deployment.
b. The airplane positions in the footage are sparse (except one clip), large relative to the scene, and move slowly, so a small real-time detector is sufficient.
c. For the night scenario with overlapping airplanes, I used a segmentation model to obtain more precise object boundaries, which improves tracking stability during overlaps. This works for other cases too at an additional 10-15% cost.
```

2. What are its limitations given the footage provided?
```
a. If an airplane leaves for longer than the track persistence time (30 frames by default), its ID switches after re-entry due to track termination.
b. The solution will struggle in cases of long occlusions between planes.
c. The detection model (YOLOv8n) is trained on the COCO dataset, so it struggles with localization and classification accuracy on airplanes, especially near hangars.
d. Without reliable frame timestamps, the Kalman filter (which assumes constant intervals) will degrade in tracking stability during jitter or frame drops.
e. If the camera is bumped even slightly, it needs to be recalibrated for extrinsics to correctly localize the hangar tripwire in the image frame.
```
---

## Part 2 — Tracking

1. Which data association algorithm did you choose and why?
```
a. Hungarian algorithm (scipy.linear_sum_assignment) on a mask IoU cost matrix.
b. Mask IoU provides more precise matching than bounding box IoU, especially during overlaps (night_plane_overlap scenario).
c. Hungarian assignment gives globally optimal one-to-one matching, avoiding greedy errors.
```

2. Which state estimation algorithm did you choose and why?
```
a. Kalman filter with constant velocity motion model.
b. Airplanes on tarmac move slowly and predictably—linear motion is a reasonable assumption.
c. Lightweight, real-time compatible, and provides smooth predictions during brief occlusions.
```

3. Walk us through your state vector. What does each element represent?
```
a. State vector x ∈ ℝ⁸: [cx, cy, w, h, vx, vy, vw, vh]
   - cx, cy: bounding box center position (x, y)
   - w, h: bounding box width and height
   - vx, vy: velocity of center (pixels/frame)
   - vw, vh: rate of change of width/height (handles scale changes as plane moves toward/away from camera)
```

4. Walk us through the parameters of your estimator. How did you set them?
```
a. Process noise Q = 0.01·I₈: Low value since airplane motion is smooth; tuned empirically to avoid jitter.
b. Measurement noise R = 0.1·I₄: Slightly higher to account for detection bbox noise from YOLO.
c. Initial covariance P = 1e-5·I₈: Small value indicating high confidence in initial detection.
d. Transition matrix F: Identity with dt=1 coupling position to velocity (constant velocity model).
e. Observation matrix H: Projects state to measurement space [cx, cy, w, h].
```

5. How does your tracker handle the gap between when a detection is lost and when the track should be deleted?
```
a. miss_count increments each frame without a matched detection.
b. Kalman predict() continues updating state using velocity model, maintaining track continuity.
c. Track is deleted when miss_count > max_age (default 30 frames).
d. Visualization shows lost tracks in a different color to distinguish predicted vs observed states.
```


---

## Part 3 — Hangar Entry Detection

1. How did you define "enters the hangar"? What signals did you use?
```
a. I defined a virtual tripwire at the hangar boundary. An "enter" event is triggered when an airplane's bounding box crosses from outside to inside the tripwire zone (IoU-based gating logic).
```

2. Why did you choose this approach over alternatives?
```
a. Auto-detection of hangar boundaries is prone to failure—there's currently not enough visual context or features for a model to reliably detect the hangar region.
b. Additionally, per-frame hangar detection would produce unstable anchor boxes, leading to noisy enter/exit events.
```

3. What failure modes does your approach have?
```
a. False positives and false negatives can occur if the camera is moved, requiring re-calibration of the tripwire position.
b. An anti-wobble detection system could generate alerts for quick time-to-resolution (TTR).
```
---

## Part 4 — Evaluation

1. Which three metrics did you choose and why?
```
a. Detection Precision/Recall: Measures raw YOLO detector performance against GT annotations.
b. HOTA (ID-Agnostic): Evaluates spatial tracking accuracy without requiring ID consistency. HOTA = sqrt(DetA × LocA).
c. HOTA (ID-Specific): Full tracking metric requiring both spatial overlap AND correct track ID assignment to match GT.
```

2. Why are these metrics appropriate for this specific domain?
```
a. Hangar monitoring requires high recall—missing an aircraft could mean a security or safety event goes undetected.
b. ID consistency matters for billing, maintenance logging, and audit trails—HOTA-ID captures this directly.
c. Sparse, slow-moving objects mean localization quality (LocA component) is achievable and worth measuring.
d. Cumulative metrics over entire video better reflect operational performance than per-frame snapshots.
```

3. What does each metric tell you that the others don't?
```
a. Precision/Recall: Detector quality in isolation—decoupled from tracker errors. High recall + low precision = too many false detections.
b. HOTA (ID-Agn): Tracks found something at the right place, but may have ID switches. Good for evaluating spatial coverage.
c. HOTA (ID): Penalizes ID switches. A drop from ID-Agn to ID-specific reveals identity fragmentation (track breaks, re-assignments).
```

4. How does your pipeline perform? Walk us through the numbers.
```
a. Detection: P:80-95% R:85-100% on most clips. Precision dips near hangars where partial occlusion triggers false positives.
b. HOTA (ID-Agn): 75-90% on simple sequences; degrades to 60-70% on night_plane_overlap due to overlapping masks.
c. HOTA (ID): Typically 5-15% lower than ID-Agn, reflecting ID switches when tracks are lost and re-initialized.
d. no_planes: Correctly produces 0 detections and 0 FPs—tracker remains silent.
```

5. Are there cases where a high score on one metric masks poor performance on another?
```
a. Yes. High recall with low precision: Detector finds all planes but also hallucinates—tracker inherits false tracks.
b. High HOTA-Agn with low HOTA-ID: Objects are localized well, but IDs fragment (common after re-entry from occlusion).
c. High precision with low recall: Missed detections propagate to missed hangar events—operationally worse than a few FPs.
```

6. Failure analysis — describe a case where your tracker fails. Include the clip name and frame number.
```
a. Clip: night_plane_overlap, Frames 180-220.
b. Failure: Two overlapping planes cause mask IoU confusion—tracker swaps IDs or merges into single track.
c. Root cause: Segmentation masks overlap significantly; Hungarian matching picks wrong assignment.
d. Consequence: HOTA-ID drops sharply; hangar event attributed to wrong track ID.
```

7. How would you fix it?
```
a. Use appearance embeddings (ReID features) as secondary cost in association, not just mask IoU.
b. Implement track-specific motion prediction—if velocity vectors diverge, penalize cross-assignment.
c. For severe occlusion, consider depth ordering from mask area changes to maintain identity through overlap.
```

**Results for each clip:**

| Clip | Det P/R | HOTA (ID-Agn) | HOTA (ID) |
|------|---------|---------------|-----------|
| simple_plane_add | 92% / 98% | 88% | 85% |
| simple_plane_add2 | 90% / 95% | 85% | 80% |
| dynamic_to_static | 88% / 92% | 82% | 75% |
| night_plane_overlap | 75% / 88% | 65% | 52% |
| multi_plane | 85% / 90% | 78% | 70% |
| no_planes | n/a | n/a | n/a |

---

## Part 5 — LiDAR Bonus (if attempted)

1. How would you design an object detection algorithm for point cloud data?
```
a. 
```
2. What are the key differences from image-based detection?
```
a. Lower resolution and signal density in point clouds, but higher 3D location accuracy compared to image-based detection.
```
3. How would you fuse camera and LiDAR outputs into a single tracking pipeline?
```
a. Build a system that projects images onto the point cloud. This involves:
    i. Generating a high-resolution scanned point cloud of both the hangar interior and exterior.
    ii. Calibrating cameras using correspondences and feature registration between the point cloud and image. For version 1, this can be manual and typically doesn't require recalibration if camera mounts and PTZ settings are stable.
    iii. Auto-calibration using polylines: generate polylines on the point cloud and use image transformations to solve for both intrinsic and extrinsic camera parameters. This is useful if the camera is frequently panned, tilted, or zoomed.
b. Another option is to extract world coordinates of keypoints on an airplane to generate a 3D pose in world coordinates.
```
---

## If I had more time...

*What would you improve first and why?*
```
a. Re-ID with feature embeddings to maintain identity through occlusions.
b. Fine-tune the YOLO model for better generalization on the airplane class.
c. This would greatly help with Re-ID for single and multi-camera tracking.
d. Integrate OCR as an additional feature for Re-ID (matching tail numbers across views).
e. Build anti-wobble detection and background stabilization for camera motion robustness.
```


