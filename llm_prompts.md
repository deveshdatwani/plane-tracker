# LLM Prompt Log

Log any prompts you submitted to AI tools (ChatGPT, Claude, Copilot, etc.) while working on this assignment.

**Tool:** Claude Opus 4.5 in GitHub Copilot

**Note:** All outputs were directly edited into the codebase via Copilot's file editing capabilities.

---

## Session Prompts

1. **Prompt:** "restructure this repository such that while loop doesn't run inside run. I want a processing module where detector runs. create a hangar_control module that tracks and detects events"
   - Created modular architecture: run.py as entry point, src/processing.py with Detector class and run_processing() frame loop, src/hangar.py with HangarControl (IOU-based event detection) and HangarControlManager (orchestrates tracker + hangar + JSON output). Separated concerns between detection, tracking, and event handling.

2. **Prompt:** "use logging instead of print everywhere"
   - Replaced all print statements with Python's logging module across the codebase for proper log levels and structured output.

3. **Prompt:** "I don't want multiple dependency for hangar and detector, consolidate modules"
   - Merged detector.py into processing.py (single detection module), merged hangar_control.py into hangar.py (single event handling module). Reduced file count and simplified import structure.

4. **Prompt:** "I want hangar tripwire to have a text that says hangar boundary, and add a gray shade to the hangar region"
   - Enhanced hangar visualization with "HANGAR BOUNDARY" text label, semi-transparent gray fill for the hangar zone, and styled boundary line.

5. **Prompt:** "do the same formatting for detections - show hits, miss, age info on track bounding boxes"
   - Added informative labels on track bounding boxes showing track ID, hits count, miss count, and age. Used consistent styling with the hangar boundary text.

6. **Prompt:** "write an OCR extractor in processing module to extract tail numbers like Nxxx from aircraft"
   - Created TailNumberExtractor class using pytesseract with multiple preprocessing approaches (grayscale, CLAHE, thresholding, scaling) and regex pattern matching for US aircraft N-numbers.

7. **Prompt:** "create a yaml called config and make detection, tracker, ocr, hangar settings configurable"
   - Created config.yaml with sections for detection (model, thresholds), tracker (IOU, max_age, min_hits), OCR (enabled, interval), hangar (cooldown, flash settings), debug, and visualization. Added src/config.py with load_config() and get_config() helpers with sensible defaults.

8. **Prompt:** "when a hangar event is detected, I want the hangar tint and boundary to flash with configurable color"
   - Implemented flash effect triggered on enter/exit events. Configured via hangar.flash_on_event, flash_color (muted yellow default), and flash_rate. Flash persists for cooldown duration.

9. **Prompt:** "I want another setting called processing_debug that draws all detections with IDs and trajectory curves"
   - Added debug.processing_debug config option. When enabled, draws dashed gray boxes around raw YOLO detections with index/class/confidence labels, plus smooth trajectory curves for each track.

10. **Prompt:** "currently, the processing debug draws trajectory of last few detections, I want it to draw traces from first detection and fit a curve that refines as new detections come in"
    - Changed to polynomial curve fitting using np.polyfit over all historic points. Curve refines as more detections accumulate, using arc-length parameterization for better handling of variable speed motion.

11. **Prompt:** "use RANSAC in the trajectory mapping to remove outlier detections"
    - Integrated scikit-learn's RANSACRegressor with polynomial features to robustly fit x(t) and y(t) curves. Outliers marked with red X markers. Added scikit-learn to requirements.txt.

12. **Prompt:** "I want previous points to be picked/dropped as new centroids come in. use size of bbox as context - if size is huge, ignore minor movements. use weighted least squares to minimize error"
    - Implemented weighted least squares where larger bboxes get higher weight (more stable detections). Movement threshold scales with bbox diagonal (3% = noise). Outliers pruned from history when residual exceeds 15% of bbox diagonal.

13. **Prompt:** "let's restructure the codebase, move tracker and drawing into a lib directory, sanity check all imports"
    - Created src/lib/ directory, moved tracker.py, drawing.py into it. Updated all import statements in run.py, hangar.py, tracker.py. Validated all imports work correctly.

14. **Prompt:** "remove all print statements, replace with logs, add logs wherever useful for debugging"
    - Added logging throughout: HangarControl/HangarControlManager init and events, PlaneTracker init/track creation/deletion, OCR extraction success, processing loop stats. Used INFO for important events, DEBUG for detailed trace.

15. **Prompt:** "utils can move inside lib too, do all sanity checks"
    - Moved utils.py to src/lib/utils.py, updated imports in run.py and hangar.py, validated all imports.

16. **Prompt:** "OCR still not working, let's debug this. can we log if something was attempted?"
    - Added detailed OCR logging: bbox coordinates and crop size on each attempt, preprocessing method used, raw OCR text output (first 5 results if no match), success with method name when tail number found.

17. **Prompt:** "yes" (to install pytesseract dependency)
    - Installed pytesseract Python package via pip. Note: tesseract-ocr system binary still needs to be installed separately.

18. **Prompt:** "when processing debug is on, I only want bounding boxed image regions displayed on frame, rest should be black (0,0,0). persist all historic boxes in the frame"
    - Modified draw_processing_debug to set frame to black, then render all historic bounding box crops at their original positions. Creates a "trail" visualization showing where objects have been detected throughout the video.

19. **Prompt:** "for smoothening trajectories, let's use last 20 points (detection centroids) instead - because if an airplane stops and goes backwards, it creates a bad trajectory"
    - Changed trajectory smoothing to use only last 20 centroids for curve fitting. Handles direction changes (stopping, reversing) much better. Lowered polynomial degree to max 3 for stability with fewer points.

20. **Prompt:** "perfect, now log all prompts that I made in this session into the llm_prompts.md. describe the tool I used in 1 line. Claude Opus 4.5 in CoPilot. skip how you used output, because all outputs were edited"
    - Created initial llm_prompts.md with all 19 prompts from the session, tool description, and note about direct code editing.

21. **Prompt:** "can you make the prompt a little more verbose? especially the first one, it depicts important info"
    - Expanded all prompt descriptions with more detail, particularly the first one explaining the full modular architecture (run.py entry point, processing.py, hangar.py with HangarControl/HangarControlManager).

22. **Prompt:** "take a look at GOALS.md. are we following all contracts for run.py? I see we are saving outputs even when I dont add output CLA in the command line. also can we add hangar event to the output?"
    - Fixed --output to be optional (default None, only writes when specified). Added hangar_events array to frame output in JSON with track_id and event_type for each enter/exit event.

23. **Prompt:** "perfect. now let's create a SOLUTION.md that describes how to run the system in various modes at the top. below I want an example GIF from simple_airplane_add2. keep it minimalistic"
    - Created SOLUTION.md with usage examples, config highlights, output format, and generated demo.gif from simple_plane_add2 video using ffmpeg.

24. **Prompt:** "the output video is incorrect. it does not have the debug info on the bounding box, like hits, age etc. also, the OCR output only says N88X when the tail number is very different. fix the above issues, also match tail number text formatting with overlay debug"
    - Made track info (ID, age, hits, miss) always visible regardless of debug.level setting. Matched tail number font to debug overlay style. Improved OCR character cleanup (O→0, I→1 in digit positions).

25. **Prompt:** "can we overlay the ground truth on this video. so I can take a look at how noisy the ground truth is. make this a configurable setting in debug"
    - Added debug.show_ground_truth config option (default false) and visualization.gt_color (magenta). Created draw_ground_truth() function in drawing.py that renders dashed magenta boxes with "GT:{track_id}" label for each ground truth annotation. Hooked into draw_debug pipeline via gt_annotations passthrough.

26. **Prompt:** "show_ground_truth is not displaying gt bboxes"
    - Fixed bug where load_annotations() returns the frames dict directly, but draw_ground_truth() was incorrectly looking for a nested "frames" key. Updated draw_ground_truth to access gt_annotations directly.

27. **Prompt:** "the section where overlay text is drawn, I want to show detection accuracy, FP and FN by matching with ground truth. Only show this info if debug show_ground_truth is true"
    - Added detection metrics (TP, FP, FN, accuracy) to the debug overlay when show_ground_truth is enabled. Implemented _compute_iou() and _compute_detection_metrics() functions with greedy IoU matching (threshold 0.5).

28. **Prompt:** "these metrics need to be cumulative, not per frame, FP should be precision and FN should be recall"
    - Changed metrics to cumulative across all frames using global _cumulative_metrics storage. Display now shows Precision and Recall instead of accuracy. Metrics reset automatically when a new video starts (frame_id goes backwards).

29. **Prompt:** "I want ID agnostic HOTA and ID specific HOTA. so total 3 kinds of metrics"
    - Implemented 3 metric sections in overlay: (1) Detection - precision/recall from raw YOLO detections vs GT, (2) HOTA (ID-Agn) - ID-agnostic HOTA using tracks vs GT with spatial match only, (3) HOTA (ID) - ID-specific HOTA requiring track ID to match GT track ID. Each section separated by "---" with heading.

---