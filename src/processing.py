from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import time
import re
import logging
# Delay importing heavy third-party libs until needed
YOLO = None
torch = None
pytesseract = None


@dataclass
class Detection:
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str = ""
    frame_id: int = -1

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


# Regex pattern for US aircraft tail numbers: N followed by digits and optional letters
# Format: N + 1-5 digits/letters, more strict pattern
TAIL_NUMBER_PATTERN = re.compile(r'N[0-9]{1,5}[A-Z]{0,2}', re.IGNORECASE)


class TailNumberExtractor:
    """Extract aircraft tail numbers (N-numbers) from bounding box regions using OCR."""
    
    def __init__(self):
        global pytesseract
        self._available = False
        try:
            import pytesseract as _pytesseract
            pytesseract = _pytesseract
            # Test that tesseract binary is available
            pytesseract.get_tesseract_version()
            self._available = True
            logging.getLogger(__name__).info("OCR: pytesseract initialized successfully")
        except ModuleNotFoundError:
            logging.getLogger(__name__).warning("OCR: pytesseract not installed. Install via `pip install pytesseract`")
        except Exception as e:
            logging.getLogger(__name__).warning(f"OCR: tesseract binary not found or error: {e}. Install tesseract-ocr.")
    
    def extract(self, frame: np.ndarray, bbox: List[float]) -> Optional[str]:
        """
        Extract tail number from the bounding box region of the frame.
        
        Args:
            frame: Full frame image (numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Tail number string if found, None otherwise
        """
        if not self._available:
            return None
        
        import cv2
        logger = logging.getLogger(__name__)
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            logger.debug(f"OCR skipped: invalid bbox ({x1},{y1})-({x2},{y2})")
            return None
        
        crop_w, crop_h = x2 - x1, y2 - y1
        logger.debug(f"OCR attempt: bbox=({x1},{y1})-({x2},{y2}), crop_size={crop_w}x{crop_h}")
        
        # Crop the bounding box region
        crop = frame[y1:y2, x1:x2]
        
        # Try multiple preprocessing approaches
        preprocessed_images = []
        
        # 1. Basic grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        preprocessed_images.append(("gray", gray))
        
        # 2. High contrast (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        preprocessed_images.append(("clahe", clahe_img))
        
        # 3. Binary threshold (for dark text on light background)
        _, thresh_dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("thresh_dark", thresh_dark))
        
        # 4. Inverted threshold (for light text on dark background)
        _, thresh_light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_images.append(("thresh_light", thresh_light))
        
        # 5. Scaled up (small text)
        scale = 2
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        preprocessed_images.append(("scaled", scaled))
        
        all_texts = []
        for preprocess_name, img in preprocessed_images:
            try:
                # Run OCR with different page segmentation modes
                for psm in [6, 7, 11, 3]:  # 6=block, 7=single line, 11=sparse, 3=auto
                    config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(img, config=config)
                    
                    # Clean up OCR output
                    text = text.upper().replace(' ', '').replace('\n', '')
                    # Fix common OCR misreads: O->0, I->1, S->5, B->8
                    
                    if text:
                        all_texts.append(f"{preprocess_name}/psm{psm}:{text}")
                    
                    # Find tail number pattern
                    matches = TAIL_NUMBER_PATTERN.findall(text)
                    if matches:
                        # Clean up the match - convert O to 0 in numeric positions
                        result = matches[0].upper()
                        # N followed by digits, so convert any O after N to 0
                        cleaned = result[0]  # Keep N
                        for i, c in enumerate(result[1:], 1):
                            if c == 'O' and i <= 5:  # In digit positions
                                cleaned += '0'
                            elif c == 'I' and i <= 5:
                                cleaned += '1'
                            else:
                                cleaned += c
                        # Ensure it starts with N and has valid format
                        if cleaned.startswith('N') and len(cleaned) >= 4:
                            logger.info(f"OCR found tail number: {cleaned} (from {preprocess_name}/psm{psm}, raw: {result})")
                            return cleaned
            except Exception as e:
                logger.debug(f"OCR attempt failed ({preprocess_name}): {e}")
                continue
        
        if all_texts:
            logger.debug(f"OCR raw outputs (no match): {all_texts[:5]}")  # Log first 5
        else:
            logger.debug("OCR: no text extracted from any preprocessing")
        
        return None


class Detector:
    def __init__(self, model_path="yolov8s-seg.pt", conf_thresh=0.4, iou_thresh=0.45, verbose=False):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.verbose = verbose
        global YOLO, torch
        try:
            if YOLO is None:
                from ultralytics import YOLO as _YOLO
                YOLO = _YOLO
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("ultralytics package is required for detection. Install via `pip install ultralytics`") from e
        try:
            if torch is None:
                import torch as _torch
                torch = _torch
        except ModuleNotFoundError:
            torch = None

        self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray):
        # Return the raw ultralytics results for downstream processing
        results = self.model(frame, conf=self.conf_thresh, iou=self.iou_thresh, verbose=self.verbose)
        return results


def run_processing(cap, detector, hangar_manager, writer=None, gt_annotations=None, no_display=False):
    import cv2
    import time

    logger = logging.getLogger(__name__)
    pipeline_times = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Starting processing loop ({total_frames} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t_start = time.perf_counter()

        # get masked frame if hangar manager provides masking helper
        if hasattr(hangar_manager, "get_masked_frame"):
            masked_frame = hangar_manager.get_masked_frame(frame)
        else:
            masked_frame = frame

        detections = detector.detect(masked_frame)

        # hand detections to hangar_control manager which will handle tracking and result generation
        hangar_manager.handle_frame(detections, frame, frame_id)

        t_end = time.perf_counter()
        pipeline_times.append(t_end - t_start)

        # optional display / writer handled by hangar_manager
        vis_frame = None
        if hasattr(hangar_manager, "draw_debug"):
            vis_frame = hangar_manager.draw_debug(frame, frame_id, detections, pipeline_times, gt_annotations)

        if writer and vis_frame is not None:
            writer.write(vis_frame)

        if not no_display and vis_frame is not None:
            cv2.imshow("SkyView - Aircraft Tracker", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info(f"User quit at frame {frame_id}")
                break

    avg_time = np.mean(pipeline_times) if pipeline_times else 0
    logger.info(f"Processing complete: {len(pipeline_times)} frames, avg {avg_time*1000:.1f}ms/frame")
    # return pipeline timings
    return pipeline_times
