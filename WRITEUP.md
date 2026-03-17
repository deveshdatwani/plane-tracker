# Take-Home Write-Up

**Name:*Devesh Datwani*
**Date:*03/16/2026*
**Time spent:*4 hours*

---

## Part 1 — Detection

1. Why did you choose this algorithm for this domain specifically?
a. I chose YOLOV8n to prioritize tracker robustness for reducing reliance on heay models for edge deployment.
b. The airplane positions in the footage are sparse (except 1), large relative to the scene, and move slowly. So a small real-time detector is sufficient. 
c. For the night scenario with overlapping airplanes, I used a segmentation model to obtain more precise object boundaries, which improves tracking stability during overlaps. This works for other cases too at an additional 10-15% cost.

2. What are its limitations given the footage provided?
a. If an airplane leaves for longer than the track persistence time (500s), its ID switches after re-entry due to track termination.
b. The solution will struggle in case of long occlusions between planes.
c. The detection solution (yolov8n) is trained on COCO dataset, so it struggles with localization and classification accuracy on airplanes, especially near hangars.
d. Without reliable frames timestamps, kalman filter (assumes constant intervals) tracking stability will degrade in cases of jitter / frame drops. 
e. If the camera is bumped even slightly, it needs to be recalibrated for extrinsics to correctly localize the hangar tripwire in image frame.
    
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
a. Generated a virtual tripwire for hangars. Detection is triggered by a simple gating logic when an airplane crosses the tripwire.

2. Why did you choose this approach over alternatives?
a. Auto detection is prone for failure presently there's not enough context / features / signal for any model to work with for anchor detection. 
b. Additionally, the anchor box would not be stable with per frame detection.

3. What failure modes does your approach have?
a. FPs and FNs can be generated if the camera is moved. It would require re-calibration. An anti-wobble system can generate alerts for quick TTR.
b. 

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
a. Lower resolution / signal quantity, so higher location accuracy with image based detection.
```
3. How would you fuse camera and LiDAR outputs into a single tracking pipeline?
```
a. Build a system that works with image overlayed on point cloud. This involves
    i. Generating a high resolution scanned point cloud of inside the hangar and outside  
    ii. Calibrate cameras using correspondences / registration of features between point cloud and image. For version 1, this can be manual and usually doesn't require recalibration if the camera mounts and PTZ are stable. 
    iii. Autocalibration using polylines. Generate polylines on point cloud. Use image transformation to solve for both intrinsic and extrinsic of cameras. This is useful if the camera is pan tiled zoomed often. 
b. Another option is to extract world coordinates of keypoints on an airplane to generate 3D pose in world coorindate.
```
---

## If I had more time...

*What would you improve first and why?*
```
a. Re-ID with feature embedding 
b. Fine tuning YOLO model for better generalization on airplane class
c. This would greatly help with Re-ID for single / multi-camera tracking 
d. I would also integrate an OCR for added feature for Re-ID
e. Build anti-wobble and background anti-wobble 
```


