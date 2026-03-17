import argparse
import json
import sys
from pathlib import Path
import logging

import cv2

TRACK_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
]


def main():
    parser = argparse.ArgumentParser(description="View annotations on video")
    parser.add_argument("name", type=str, help="Video name (e.g. simple_plane_add)")
    parser.add_argument("--save-video", type=str, default=None, help="Path to save output video")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger(__name__)

    video_path = Path(f"data/{args.name}.mp4")
    ann_path = Path(f"annotations/{args.name}.json")

    assert video_path.exists(), f"Video not found: {video_path}"
    assert ann_path.exists(), f"Annotations not found: {ann_path}"

    with open(ann_path, "r") as f:
        raw = json.load(f)
    annotations = raw.get("frames", {})

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_frame_ids = sorted(int(k) for k in annotations.keys())
    total_annotated = len(annotated_frame_ids)

    logger.info(f"Video: {video_path.name} | {width}x{height} @ {fps:.1f}fps | {total_frames} frames")
    logger.info(f"Annotations: {total_annotated} frames annotated")
    if annotated_frame_ids:
        logger.info(f"  Range: frame {annotated_frame_ids[0]} to {annotated_frame_ids[-1]}")
    if not args.no_display:
        logger.info("Controls: Space = play/pause, Left/Right = step, Q/ESC = quit")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
        logger.info(f"Recording to {args.save_video}")

    frame_idx = 0
    playing = False

    def get_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        return frame if ret else None

    frame = get_frame(frame_idx)
    if frame is None:
        logger.error("ERROR: Could not read video")
        return

    while True:
        vis = frame.copy()
        frame_key = str(frame_idx)
        frame_anns = annotations.get(frame_key, [])

        # Draw annotations
        for ann in frame_anns:
            tid = ann["track_id"]
            x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
            color = TRACK_COLORS[tid % len(TRACK_COLORS)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"GT ID:{tid} ({ann.get('class', 'aircraft')})"
            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Info overlay
        has_ann = "YES" if frame_anns else "no"
        info = f"Frame {frame_idx}/{total_frames - 1} | Annotations: {has_ann} ({len(frame_anns)} objects)"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Annotation coverage bar
        bar_y = height - 10
        for aid in annotated_frame_ids:
            bx = int(aid / total_frames * width)
            cv2.line(vis, (bx, bar_y - 4), (bx, bar_y + 4), (0, 255, 0), 1)
        # Current position marker
        cx = int(frame_idx / total_frames * width)
        cv2.line(vis, (cx, bar_y - 8), (cx, bar_y + 8), (0, 0, 255), 2)

        if writer:
            writer.write(vis)

        if not args.no_display:
            cv2.imshow("Annotation Viewer", vis)
            wait_ms = int(1000 / fps) if playing else 0
            key = cv2.waitKey(wait_ms) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                playing = not playing
            elif key in (ord('d'), 83):  # Right
                frame_idx = min(frame_idx + 1, total_frames - 1)
                frame = get_frame(frame_idx)
                if frame is None:
                    break
                continue
            elif key in (ord('a'), 81):  # Left
                frame_idx = max(frame_idx - 1, 0)
                frame = get_frame(frame_idx)
                if frame is None:
                    break
                continue

        if playing:
            frame_idx += 1
            if frame_idx >= total_frames:
                break
            frame = get_frame(frame_idx)
            if frame is None:
                break
        elif args.no_display:
            frame_idx += 1
            if frame_idx >= total_frames:
                break
            frame = get_frame(frame_idx)
            if frame is None:
                break

    cap.release()
    if writer:
        writer.release()
        logger.info(f"Video saved to {args.save_video}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
