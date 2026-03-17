# Take-Home Write-Up

**Name:*Devesh Datwani*
**Date:*03/16/2026*
**Time spent:*4 hours*

---

## Part 1 — Detection

1. Why did you choose this algorithm for this domain specifically?
```
a. I chose YOLOV8n to prioritize tracker robustness for reducing reliance on heay models for edge deployment.
b. The airplane positions in the footage are sparse (except 1), large relative to the scene, and move slowly. So a small real-time detector is sufficient. 
c. For the night scenario with overlapping airplanes, I used a segmentation model to obtain more precise object boundaries, which improves tracking stability during overlaps. This works for other cases too at an additional 10-15% cost.
```

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
a. I used my boilerplate code for multi object tracking for initial tests. 
b. It generalised well over sequences in the footage, after fine tuning noise parameters for process and measurements steps. 
c. Since the ground truths were not robust, I approach the problem with extensive visualizations and HOTA, class agnostic HOTA, precision and recall. 
d. Precision and recall can give us a good estimate of 2D detection, while HOTA and class agnostic HOTA give us not perfect, but some understanding of tracker performance.
```

2. Which state estimation algorithm did you choose and why?

3. Walk us through your state vector. What does each element represent?

4. Walk us through the parameters of your estimator. How did you set them?

5. How does your tracker handle the gap between when a detection is lost and when the track should be deleted?
a. Drawing debug prediction boxes and traces in different colros


---

## Part 3 — Hangar Entry Detection

1. How did you define "enters the hangar"? What signals did you use?
a. I generated a virtual tripwire for hangars. Detection is triggered by a simple gating logic when an airplane crosses the tripwire.

2. Why did you choose this approach over alternatives?
a. Auto detection is prone for failure presently there's not enough context / features / signal for any model to work with for anchor detection. 
b. Additionally, the anchor box would not be stable with per frame detection.

3. What failure modes does your approach have?
a. FPs and FNs can be generated if the camera is moved. It would require re-calibration. An anti-wobble system can generate alerts for quick TTR.
b. 

---

## Part 4 — Evaluation

1. Which three metrics did you choose and why?
2. Why are these metrics appropriate for this specific domain?
3. What does each metric tell you that the others don't?
4. How does your pipeline perform? Walk us through the numbers.
5. Are there cases where a high score on one metric masks poor performance on another?
6. Failure analysis — describe a case where your tracker fails. Include the clip name and frame number.
7. How would you fix it?

**Results for each clip:**

| Clip | Metric 1 | Metric 2 | Metric 3 |
|------|----------|----------|----------|
| simple_plane_add | | | |
| simple_plane_add2 | | | |
| dynamic_to_static | | | |
| night_plane_overlap | | | |
| multi_plane | | | |
| no_planes | | | |

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


