#!/usr/bin/env python3
"""Build one-image temporal-canvas VLM samples from annotated sequential frames.

This offline tool is intentionally limited to the current Gazebo test format:
each input PNG already contains one bright-green ByteTrack bounding box.  It
groups ordered frames into non-overlapping four-frame clips, recovers that box,
and writes the final RGB frame overlaid with an observed oldest-to-newest trail.
The output JSON keeps the existing ``[{"id": ..., "state": ...}]`` target
contract while replacing four ``<image>`` placeholders with one canvas image.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


# Gazebo's annotation uses a saturated green box.  The range deliberately
# excludes grey world geometry while accepting small PNG compression changes.
GREEN_LOWER_HSV = np.array((45, 120, 120), dtype=np.uint8)
GREEN_UPPER_HSV = np.array((85, 255, 255), dtype=np.uint8)
TRAIL_COLORS_BGR = ((190, 190, 190), (255, 180, 0), (0, 220, 255), (0, 0, 255))
PROMPT = (
    '<image>\nYou are given one temporal trajectory canvas from a robot-view scene. '
    'The background is the newest RGB frame at t=1.5 seconds. Each visible person '
    'has a ByteTrack ID label. For each ID, numbered trail points show observed '
    'positions in temporal order: 1 = t=0.0s, 2 = t=0.5s, 3 = t=1.0s, '
    '4 = t=1.5s (now). The arrow points from the older position to the newest '
    'position. Use only IDs visible in the newest frame. Return only a JSON array '
    'with one object containing "id" and "state" for each visible person.')


def parse_arguments() -> argparse.Namespace:
    """Read explicit dataset paths and semantic labels from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-dir', type=Path, required=True,
                        help='Directory containing chronologically named frame_*.png images.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for canvas PNGs and temporal_canvas_train.json.')
    parser.add_argument('--track-id', type=int, required=True,
                        help='ByteTrack ID represented by the bright-green input box.')
    parser.add_argument('--state', required=True,
                        choices=('crossing', 'approaching', 'straight', 'talking'),
                        help='Ground-truth semantic state shared by this test sequence.')
    parser.add_argument('--frames-per-clip', type=int, default=4,
                        help='Sequential frames per canvas; this VLM contract requires 4.')
    parser.add_argument('--stride', type=int, default=None,
                        help='Frames to advance after a clip; default keeps clips non-overlapping.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow an existing output directory.')
    return parser.parse_args()


def find_green_bbox(image: np.ndarray, path: Path) -> Tuple[int, int, int, int]:
    """Recover one annotated ByteTrack box as ``x1, y1, x2, y2`` pixel bounds.

    The input training frames already have the green ByteTrack overlay, so
    recovering its largest rectangle preserves exactly the same identity cue the
    label JSON refers to.  This avoids silently inventing a new tracker ID.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER_HSV, GREEN_UPPER_HSV)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= 40 and height >= 80:
            candidates.append((width * height, x, y, width, height))
    if not candidates:
        raise ValueError(f'No bright-green ByteTrack box found in {path}')
    _, x, y, width, height = max(candidates)
    return x, y, x + width, y + height


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Return the image-space center used for an observed temporal trail."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def draw_label(image: np.ndarray, point: Tuple[int, int], label: str,
               color: Tuple[int, int, int]) -> None:
    """Draw one numbered temporal point with a contrasting outline for resizing."""
    cv2.circle(image, point, 8, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, point, 5, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (point[0] + 10, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (point[0] + 10, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def build_canvas(frames: Sequence[np.ndarray], paths: Sequence[Path]) -> np.ndarray:
    """Render four observed green-box centers over the newest RGB frame.

    Only history is drawn: point 1 is the oldest input frame and point 4 is the
    current frame.  The final arrow is not a future prediction, which prevents
    target leakage into the VLM training image.
    """
    boxes = [find_green_bbox(frame, path) for frame, path in zip(frames, paths)]
    centers = [bbox_center(box) for box in boxes]
    canvas = frames[-1].copy()
    for previous, current, color in zip(centers, centers[1:], TRAIL_COLORS_BGR[1:]):
        cv2.line(canvas, previous, current, color, 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, centers[-2], centers[-1], TRAIL_COLORS_BGR[-1],
                    3, cv2.LINE_AA, tipLength=0.24)
    for point_index, (point, color) in enumerate(zip(centers, TRAIL_COLORS_BGR), start=1):
        draw_label(canvas, point, str(point_index), color)
    # The legend is part of the learned visual contract, not decoration.  A
    # solid panel keeps it legible after Qwen resizes the full camera scene.
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 50), (24, 24, 24), thickness=-1)
    cv2.putText(canvas, 'Temporal trail: 1 (oldest) -> 4 (now)', (12, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, '1=1.5s ago   2=1.0s   3=0.5s   4=now', (12, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    """Generate canvas PNGs plus an old-style conversations JSON training file."""
    args = parse_arguments()
    if args.frames_per_clip != 4:
        raise ValueError('frames-per-clip must remain 4 for the current temporal contract')
    stride = args.frames_per_clip if args.stride is None else args.stride
    if stride < 1:
        raise ValueError('stride must be positive')
    frame_paths = sorted(args.frames_dir.glob('frame_*.png'))
    if len(frame_paths) < args.frames_per_clip:
        raise ValueError(f'Need at least {args.frames_per_clip} frames, found {len(frame_paths)}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{args.output_dir} already exists; use --overwrite to reuse it')
    canvases_dir = args.output_dir / 'canvas'
    canvases_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for first_index in range(0, len(frame_paths) - args.frames_per_clip + 1, stride):
        clip_paths = frame_paths[first_index:first_index + args.frames_per_clip]
        frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in clip_paths]
        if any(frame is None for frame in frames):
            missing = [str(path) for path, frame in zip(clip_paths, frames) if frame is None]
            raise ValueError(f'Cannot read image(s): {missing}')
        canvas = build_canvas(frames, clip_paths)
        clip_number = len(records) + 1
        canvas_name = f'clip_{clip_number:04d}.png'
        canvas_path = canvases_dir / canvas_name
        if not cv2.imwrite(str(canvas_path), canvas):
            raise OSError(f'Cannot write {canvas_path}')
        records.append({
            'id': f'clip_{clip_number:04d}',
            'image': [str(Path('canvas') / canvas_name)],
            'conversations': [
                {'from': 'human', 'value': PROMPT},
                {'from': 'gpt', 'value': [{'id': args.track_id, 'state': args.state}]},
            ],
        })

    dataset_path = args.output_dir / 'temporal_canvas_train.json'
    dataset_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Created {len(records)} non-overlapping four-frame clips.')
    print(f'Canvas images: {canvases_dir}')
    print(f'Training JSON: {dataset_path}')


if __name__ == '__main__':
    main()
