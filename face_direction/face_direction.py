#!/usr/bin/env python3
"""Detect faces and report each person's head direction from webcam/video.

This single file contains the SCRFD face detector, ONNX head-pose inference,
direction classification, drawing, and CLI. It is adapted from:
https://github.com/yakhyo/head-pose-estimation

MIT License
Copyright (c) 2024 Yakhyokhuja Valikhujaev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # Give a short, actionable error instead of a long traceback.
    raise SystemExit(
        "Thiếu onnxruntime. Chạy: python3 -m pip install -r requirements.txt"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POSE_MODEL = BASE_DIR / "models" / "mobilenetv2.onnx"
DEFAULT_FACE_MODEL = BASE_DIR / "models" / "det_10g.onnx"

DIRECTION_VI = {
    "forward": "chính diện",
    "left": "trái",
    "right": "phải",
    "up": "lên",
    "down": "xuống",
    "up_left": "trái và lên",
    "up_right": "phải và lên",
    "down_left": "trái và xuống",
    "down_right": "phải và xuống",
}

DISPLAY_LABEL = {
    "forward": "CHINH DIEN",
    "left": "TRAI",
    "right": "PHAI",
    "up": "LEN",
    "down": "XUONG",
    "up_left": "TRAI + LEN",
    "up_right": "PHAI + LEN",
    "down_left": "TRAI + XUONG",
    "down_right": "PHAI + XUONG",
}


@dataclass(frozen=True)
class FaceDirection:
    """Result for one face. Angles are in degrees."""

    bbox: tuple[int, int, int, int]
    face_score: float
    pitch: float
    yaw: float
    roll: float
    direction: str
    direction_vi: str

    def to_dict(self) -> dict:
        result = asdict(self)
        result["bbox"] = list(self.bbox)
        return result


def classify_direction(
    yaw: float,
    pitch: float,
    yaw_threshold: float = 20.0,
    pitch_threshold: float = 15.0,
    person_relative: bool = True,
) -> tuple[str, str]:
    """Map angles to 9 directions; roll is tilt and is not a look direction.

    By default left/right mean the person's own left/right. Set
    ``person_relative=False`` if labels should mean left/right on the image.
    """

    values = (yaw, pitch, yaw_threshold, pitch_threshold)
    if not np.all(np.isfinite(values)):
        raise ValueError("Góc và ngưỡng phải là số hữu hạn")
    if yaw_threshold <= 0 or pitch_threshold <= 0:
        raise ValueError("Ngưỡng yaw/pitch phải lớn hơn 0")

    horizontal = ""
    if yaw > yaw_threshold:
        horizontal = "right" if person_relative else "left"
    elif yaw < -yaw_threshold:
        horizontal = "left" if person_relative else "right"

    vertical = ""
    if pitch > pitch_threshold:
        vertical = "up"
    elif pitch < -pitch_threshold:
        vertical = "down"

    if horizontal and vertical:
        direction = f"{vertical}_{horizontal}"
    else:
        direction = horizontal or vertical or "forward"
    return direction, DIRECTION_VI[direction]


def _ort_providers() -> list[str]:
    available = ort.get_available_providers()
    return [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]


def _make_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    providers = _ort_providers()
    if providers:
        return ort.InferenceSession(str(model_path), providers=providers)
    return ort.InferenceSession(str(model_path))


def _distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = np.maximum(points[:, 0] - distance[:, 0], 0)
    y1 = np.maximum(points[:, 1] - distance[:, 1], 0)
    x2 = np.maximum(points[:, 0] + distance[:, 2], 0)
    y2 = np.maximum(points[:, 1] + distance[:, 3], 0)
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_keypoints(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    coordinates = []
    for index in range(0, distance.shape[1], 2):
        coordinates.append(points[:, index % 2] + distance[:, index])
        coordinates.append(points[:, index % 2 + 1] + distance[:, index + 1])
    return np.stack(coordinates, axis=-1)


class SCRFD:
    """SCRFD ONNX face detector used by the reference repository."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (640, 640),
        confidence: float = 0.5,
        nms_iou: float = 0.4,
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("face confidence phải nằm trong (0, 1]")
        self.input_size = input_size
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.strides = (8, 16, 32)
        self.num_anchors = 2
        self.center_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self.session = _make_session(Path(model_path).expanduser().resolve())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _forward(self, image: np.ndarray):
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 128.0,
            size=(image.shape[1], image.shape[0]),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        scores_list, bboxes_list, keypoints_list = [], [], []

        for level, stride in enumerate(self.strides):
            scores = outputs[level]
            bbox_predictions = outputs[level + 3] * stride
            keypoint_predictions = outputs[level + 6] * stride
            height, width = blob.shape[2] // stride, blob.shape[3] // stride
            cache_key = (height, width, stride)
            centers = self.center_cache.get(cache_key)
            if centers is None:
                centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1)
                centers = (centers.astype(np.float32) * stride).reshape(-1, 2)
                centers = np.repeat(centers[:, None, :], self.num_anchors, axis=1).reshape(-1, 2)
                if len(self.center_cache) < 100:
                    self.center_cache[cache_key] = centers

            positive = np.where(scores >= self.confidence)[0]
            scores_list.append(scores[positive])
            bboxes_list.append(_distance_to_bbox(centers, bbox_predictions)[positive])
            keypoints = _distance_to_keypoints(centers, keypoint_predictions)
            keypoints_list.append(keypoints.reshape(keypoints.shape[0], -1, 2)[positive])

        return scores_list, bboxes_list, keypoints_list

    @staticmethod
    def _nms(detections: np.ndarray, threshold: float) -> list[int]:
        x1, y1, x2, y2, scores = detections.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            index = order[0]
            keep.append(int(index))
            intersection_x1 = np.maximum(x1[index], x1[order[1:]])
            intersection_y1 = np.maximum(y1[index], y1[order[1:]])
            intersection_x2 = np.minimum(x2[index], x2[order[1:]])
            intersection_y2 = np.minimum(y2[index], y2[order[1:]])
            width = np.maximum(0.0, intersection_x2 - intersection_x1 + 1)
            height = np.maximum(0.0, intersection_y2 - intersection_y1 + 1)
            intersection = width * height
            overlap = intersection / (areas[index] + areas[order[1:]] - intersection)
            order = order[np.where(overlap <= threshold)[0] + 1]
        return keep

    def detect(self, image: np.ndarray, max_faces: int = 0):
        target_width, target_height = self.input_size
        image_ratio = image.shape[0] / image.shape[1]
        model_ratio = target_height / target_width
        if image_ratio > model_ratio:
            resized_height = target_height
            resized_width = int(resized_height / image_ratio)
        else:
            resized_width = target_width
            resized_height = int(resized_width * image_ratio)

        scale = resized_height / image.shape[0]
        resized = cv2.resize(image, (resized_width, resized_height))
        detector_input = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        detector_input[:resized_height, :resized_width] = resized
        score_parts, bbox_parts, keypoint_parts = self._forward(detector_input)

        scores = np.vstack(score_parts).reshape(-1)
        bboxes = np.vstack(bbox_parts) / scale
        keypoints = np.vstack(keypoint_parts) / scale
        order = scores.argsort()[::-1]
        detections = np.hstack((bboxes, scores[:, None])).astype(np.float32)[order]
        keypoints = keypoints[order]
        keep = self._nms(detections, self.nms_iou)
        detections, keypoints = detections[keep], keypoints[keep]

        if 0 < max_faces < len(detections):
            areas = (detections[:, 2] - detections[:, 0]) * (
                detections[:, 3] - detections[:, 1]
            )
            selected = np.argsort(areas)[::-1][:max_faces]
            detections, keypoints = detections[selected], keypoints[selected]
        return detections, keypoints


class HeadPoseONNX:
    """Return pitch, yaw and roll in degrees from a cropped BGR face."""

    def __init__(self, model_path: str | Path) -> None:
        self.session = _make_session(Path(model_path).expanduser().resolve())
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if len(shape) != 4 or not all(isinstance(value, int) for value in shape[2:]):
            raise ValueError(f"Input model head-pose không hợp lệ: {shape}")
        self.input_name = model_input.name
        self.input_size = (shape[3], shape[2])
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def estimate(self, face_bgr: np.ndarray) -> tuple[float, float, float]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(face_rgb, self.input_size).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        rotation = np.asarray(outputs[0], dtype=np.float32)
        if rotation.shape != (1, 3, 3) or not np.all(np.isfinite(rotation)):
            raise RuntimeError(f"Output model head-pose không hợp lệ: {rotation.shape}")

        sy = np.sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
        singular = sy < 1e-6
        pitch = np.where(
            singular,
            np.arctan2(-rotation[:, 1, 2], rotation[:, 1, 1]),
            np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]),
        )
        yaw = np.arctan2(-rotation[:, 2, 0], sy)
        roll = np.where(
            singular,
            np.zeros_like(sy),
            np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]),
        )
        angles = np.degrees(np.stack([pitch, yaw, roll], axis=1))[0]
        return float(angles[0]), float(angles[1]), float(angles[2])


def _expanded_bbox(
    bbox: Sequence[float], image_shape: Sequence[int], factor: float = 0.2
) -> tuple[int, int, int, int] | None:
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = map(float, bbox[:4])
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return None
    # Match the reference code: horizontal padding uses face height and vice versa.
    pad_x, pad_y = factor * height, factor * width
    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(image_width, int(np.ceil(x2 + pad_x)))
    y2 = min(image_height, int(np.ceil(y2 + pad_y)))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


class FaceDirectionDetector:
    """Reusable API: call ``detect(frame_bgr)`` to get all face directions."""

    def __init__(
        self,
        pose_model: str | Path = DEFAULT_POSE_MODEL,
        face_model: str | Path = DEFAULT_FACE_MODEL,
        yaw_threshold: float = 20.0,
        pitch_threshold: float = 15.0,
        face_confidence: float = 0.5,
        max_faces: int = 0,
        min_face_size: int = 48,
        person_relative: bool = True,
    ) -> None:
        if max_faces < 0:
            raise ValueError("max_faces phải lớn hơn hoặc bằng 0")
        if min_face_size <= 0:
            raise ValueError("min_face_size phải lớn hơn 0")
        self.pose = HeadPoseONNX(pose_model)
        self.face = SCRFD(face_model, confidence=face_confidence)
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.max_faces = max_faces
        self.min_face_size = min_face_size
        self.person_relative = person_relative

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDirection]:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            raise ValueError("frame_bgr phải là ảnh NumPy HxWx3")
        if frame_bgr.size == 0 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr phải là ảnh BGR 3 kênh không rỗng")

        detections, _ = self.face.detect(frame_bgr, self.max_faces)
        results = []
        for detection in detections:
            box = _expanded_bbox(detection[:4], frame_bgr.shape, factor=0.0)
            crop_box = _expanded_bbox(detection[:4], frame_bgr.shape, factor=0.2)
            if box is None or crop_box is None:
                continue
            if min(box[2] - box[0], box[3] - box[1]) < self.min_face_size:
                continue
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
            face_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            pitch, yaw, roll = self.pose.estimate(face_crop)
            direction, direction_vi = classify_direction(
                yaw,
                pitch,
                self.yaw_threshold,
                self.pitch_threshold,
                self.person_relative,
            )
            results.append(
                FaceDirection(
                    bbox=box,
                    face_score=float(detection[4]),
                    pitch=pitch,
                    yaw=yaw,
                    roll=roll,
                    direction=direction,
                    direction_vi=direction_vi,
                )
            )
        return results

    def annotate(
        self,
        frame_bgr: np.ndarray,
        results: Sequence[FaceDirection],
        mirrored: bool = False,
    ) -> np.ndarray:
        """Draw results; ``mirrored`` flips the preview without reversing text."""

        output = cv2.flip(frame_bgr, 1) if mirrored else frame_bgr.copy()
        image_width = frame_bgr.shape[1]
        for result in results:
            x1, y1, x2, y2 = result.bbox
            if mirrored:
                x1, x2 = image_width - x2, image_width - x1
            color = (60, 200, 60) if result.direction == "forward" else (0, 165, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            _draw_axis(
                output,
                result.yaw,
                result.pitch,
                result.roll,
                result.bbox,
                mirror_width=image_width if mirrored else None,
            )
            label = (
                f"{DISPLAY_LABEL[result.direction]} | "
                f"Y:{result.yaw:+.1f} P:{result.pitch:+.1f} R:{result.roll:+.1f}"
            )
            label_y = max(20, y1 - 8)
            cv2.putText(
                output,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return output


def _draw_axis(
    image: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    bbox: Sequence[int],
    mirror_width: int | None = None,
) -> None:
    yaw, pitch, roll = np.radians([-yaw, pitch, roll])
    x1, y1, x2, y2 = bbox
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
    size = min(x2 - x1, y2 - y1) * 0.5
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    axis_x = (
        int(size * cos_yaw * cos_roll + center_x),
        int(size * (cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw) + center_y),
    )
    axis_y = (
        int(-size * cos_yaw * sin_roll + center_x),
        int(size * (cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll) + center_y),
    )
    axis_z = (
        int(size * sin_yaw + center_x),
        int(-size * cos_yaw * sin_pitch + center_y),
    )
    center = (center_x, center_y)
    if mirror_width is not None:
        mirror = lambda point: (mirror_width - 1 - point[0], point[1])
        center, axis_x, axis_y, axis_z = map(mirror, (center, axis_x, axis_y, axis_z))
    cv2.line(image, center, axis_x, (0, 0, 255), 2)
    cv2.line(image, center, axis_y, (0, 255, 0), 2)
    cv2.line(image, center, axis_z, (255, 0, 0), 2)


def _video_source(value: str) -> int | str:
    return int(value) if value.strip().isdigit() else value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nhận diện mặt và hướng đầu người")
    parser.add_argument("--source", default="0", help="Camera index hoặc đường dẫn video")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL))
    parser.add_argument("--face-model", default=str(DEFAULT_FACE_MODEL))
    parser.add_argument("--yaw-threshold", type=float, default=20.0)
    parser.add_argument("--pitch-threshold", type=float, default=15.0)
    parser.add_argument("--face-confidence", type=float, default=0.5)
    parser.add_argument("--max-faces", type=int, default=0, help="0 = tất cả khuôn mặt")
    parser.add_argument("--min-face-size", type=int, default=48, help="Cạnh mặt nhỏ nhất (pixel)")
    parser.add_argument("--image-relative", action="store_true", help="Trái/phải theo khung ảnh")
    parser.add_argument("--mirror", action="store_true", help="Chỉ lật ảnh hiển thị")
    parser.add_argument("--output", help="Lưu video kết quả")
    parser.add_argument("--no-view", action="store_true", help="Không mở cửa sổ")
    parser.add_argument("--jsonl", action="store_true", help="In JSON ở mọi frame")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        detector = FaceDirectionDetector(
            pose_model=args.pose_model,
            face_model=args.face_model,
            yaw_threshold=args.yaw_threshold,
            pitch_threshold=args.pitch_threshold,
            face_confidence=args.face_confidence,
            max_faces=args.max_faces,
            min_face_size=args.min_face_size,
            person_relative=not args.image_relative,
        )
        capture = cv2.VideoCapture(_video_source(args.source))
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được camera/video: {args.source}")

        writer = None
        frame_number = 0
        last_directions = None
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                results = detector.detect(frame)
                annotated = detector.annotate(frame, results, mirrored=args.mirror)

                if args.jsonl:
                    print(
                        json.dumps(
                            {
                                "frame": frame_number,
                                "status": "ok" if results else "no_face",
                                "faces": [result.to_dict() for result in results],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    directions = tuple(result.direction for result in results)
                    if directions != last_directions:
                        if results:
                            text = "; ".join(
                                f"mặt {index}: {result.direction_vi} "
                                f"(yaw={result.yaw:+.1f}°, pitch={result.pitch:+.1f}°, "
                                f"roll={result.roll:+.1f}°)"
                                for index, result in enumerate(results, 1)
                            )
                            print(text, flush=True)
                        else:
                            print("Không phát hiện khuôn mặt", flush=True)
                        last_directions = directions

                if args.output:
                    if writer is None:
                        output_path = Path(args.output).expanduser()
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        fps = capture.get(cv2.CAP_PROP_FPS)
                        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            float(fps),
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Không tạo được file: {output_path}")
                    writer.write(annotated)

                if not args.no_view:
                    cv2.imshow("Face direction - nhan Q de thoat", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if not args.no_view:
                cv2.destroyAllWindows()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
