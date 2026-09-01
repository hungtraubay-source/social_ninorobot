#!/usr/bin/env python3
"""Build one-image four-frame contact-sheet VLM samples from sequential frames.

The VLM receives one 2x2 raster image instead of four independent image inputs.
The tiles are retained as real camera frames (not synthetic motion symbols), with
their chronological timestamps written in a compact header.  This script is an
offline data-preparation tool; ROS runtime must render the identical layout before
passing an image to a contact-sheet fine-tuned model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


PROMPT = (
    '<image>\nYou are given one contact sheet containing 4 sequential robot-view '
    'frames from the same short video clip. Read the tiles in temporal order: top '
    'left = t=0.0s, top right = t=0.5s, bottom left = t=1.0s, bottom right = '
    't=1.5s (now). The tracked person ID is printed in every tile header. Compare '
    'that same ID across the four real frames. Return only a JSON array with one object '
    'containing "id" and "state" for each person visible in the newest frame.')


def parse_arguments() -> argparse.Namespace:
    """Read explicit source, output, timing, and ground-truth label settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-dir', type=Path, required=True,
                        help='Directory containing chronologically named frame_*.png images.')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for contact sheets and contact_sheet_train.json.')
    parser.add_argument('--track-id', type=int, required=True,
                        help='ByteTrack ID whose ground-truth label is supplied for this sequence.')
    parser.add_argument('--state', required=True,
                        choices=('crossing', 'approaching', 'straight', 'talking'),
                        help='Ground-truth semantic state shared by this source sequence.')
    parser.add_argument('--frame-interval-sec', type=float, default=0.5,
                        help='Real elapsed time between source frames, in seconds.')
    parser.add_argument('--stride', type=int, default=None,
                        help='Frames to advance after a clip; default creates non-overlapping clips.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow an existing output directory to be reused.')
    return parser.parse_args()


def tile_time_label(index: int, interval_sec: float) -> str:
    """Return an unambiguous timestamp label for one zero-based tile index."""
    return f't={index * interval_sec:.1f}s' + (' (now)' if index == 3 else '')


def render_tile(frame: np.ndarray, index: int, interval_sec: float, track_id: int) -> np.ndarray:
    """Resize a real RGB frame and overlay only its temporal position label.

    A 320x240 tile makes a 2x2 contact sheet 640x480, matching the original
    camera image dimensions.  Keeping this fixed avoids a fourfold visual-token
    increase caused by simply concatenating full-resolution input frames.  The
    source overlay's ID is often too small after resizing, so the header repeats
    the known target ByteTrack ID in a legible, high-contrast form.
    """
    tile = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (320, 29), (18, 18, 18), thickness=-1)
    cv2.putText(tile, tile_time_label(index, interval_sec), (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    identity_label = f'ID: {track_id}'
    identity_width = cv2.getTextSize(identity_label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
    cv2.putText(tile, identity_label, (312 - identity_width, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return tile


def build_contact_sheet(frames: Sequence[np.ndarray], interval_sec: float, track_id: int) -> np.ndarray:
    """Place four chronological real frames in reading order on one image.

    The layout is top-left, top-right, bottom-left, bottom-right.  This ordering
    is stated both in the pixel labels and in the training/runtime prompt so the
    model sees one stable temporal input contract.
    """
    if len(frames) != 4:
        raise ValueError(f'Expected four frames, received {len(frames)}')
    tiles = [render_tile(frame, index, interval_sec, track_id)
             for index, frame in enumerate(frames)]
    return np.vstack((np.hstack((tiles[0], tiles[1])), np.hstack((tiles[2], tiles[3]))))


def main() -> None:
    """Write one contact sheet and one single-image training record per four frames."""
    args = parse_arguments()
    if args.frame_interval_sec <= 0:
        raise ValueError('frame-interval-sec must be positive')
    stride = 4 if args.stride is None else args.stride
    if stride < 1:
        raise ValueError('stride must be positive')
    frame_paths = sorted(args.frames_dir.glob('frame_*.png'))
    if len(frame_paths) < 4:
        raise ValueError(f'Need at least 4 frames, found {len(frame_paths)}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f'{args.output_dir} already exists; use --overwrite to reuse it')

    sheets_dir = args.output_dir / 'contact_sheet'
    sheets_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for first_index in range(0, len(frame_paths) - 3, stride):
        clip_paths = frame_paths[first_index:first_index + 4]
        frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in clip_paths]
        if any(frame is None for frame in frames):
            unreadable = [str(path) for path, frame in zip(clip_paths, frames) if frame is None]
            raise ValueError(f'Cannot read image(s): {unreadable}')
        sheet = build_contact_sheet(frames, args.frame_interval_sec, args.track_id)
        clip_number = len(records) + 1
        sheet_name = f'clip_{clip_number:04d}.png'
        sheet_path = sheets_dir / sheet_name
        if not cv2.imwrite(str(sheet_path), sheet):
            raise OSError(f'Cannot write {sheet_path}')
        records.append({
            'id': f'clip_{clip_number:04d}',
            'image': [str(Path('contact_sheet') / sheet_name)],
            'conversations': [
                {'from': 'human', 'value': PROMPT},
                {'from': 'gpt', 'value': [{'id': args.track_id, 'state': args.state}]},
            ],
        })

    dataset_path = args.output_dir / 'contact_sheet_train.json'
    dataset_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Created {len(records)} non-overlapping four-frame contact sheets.')
    print(f'Contact sheets: {sheets_dir}')
    print(f'Training JSON: {dataset_path}')


if __name__ == '__main__':
    main()
