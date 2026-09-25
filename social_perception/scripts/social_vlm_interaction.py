#!/usr/bin/env python3
"""Classify per-person temporal social states with a 4-frame rolling buffer.

This node connects Block-B perception with the fine-tuned Qwen2-VL LoRA model.
It subscribes to Block B's synchronized output topics:
  - ``sensor_msgs/Image``: raw camera frames.
  - ``social_perception/PeopleObservations``: 2D bounding boxes with ByteTrack IDs.
  - ``social_perception/People``: validated 3D map/odom localized positions.

Workflow & Design Decisions:
1. Grounding & Identity Binding:
   Each detected track is assigned a distinct bounding-box color and an explicit
   label 'ID <id>' rendered on the frame. This allows the VLM to identify persons
   either by color (e.g. 'red', 'green') or by numeric ID ('id': 1).
2. Sim-Time Rolling Buffer:
   Maintains a 4-frame rolling buffer strictly sampled at fixed intervals (default 0.5s)
   using ROS simulation time (header stamp). Handles simulator clock resets gracefully.
3. Signature Continuity:
   Snapshots are enqueued only when track identities remain consistent across
   all 4 frames and the inference worker is not busy.
4. Video-Structure Qwen Inference & Token Confidence:
   Frames are packaged as a single temporal video sequence for Qwen2-VL.
   Geometric-mean token probability is calculated from output logits to provide
   a calibrated confidence score inside the raw JSON response.
5. Deterministic Safety Contract:
   Output topic ``/social_perception/vlm_person_states`` carries semantic labels only
   (e.g., 'talking', 'waiting', 'walking'/'crossing'). Position, velocity, and costmap
   geometry remain strictly owned by deterministic upstream modules.
"""

import json
import math
import os
import queue
import re
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from social_perception.msg import (
    People,
    PeopleObservations,
    VlmPersonState,
    VlmPersonStates,
)

# Palette of distinct BGR colors for multi-person tracking and visualization.
PALETTE_BGR = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

COLOR_NAMES = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']

STATE_VALUE_PATTERN = re.compile(
    r'"state"\s*:\s*"(?P<state>[a-zA-Z_]+)"',
    re.IGNORECASE,
)


def stamp_ns(header) -> int:
    """Return a ROS timestamp as an exact integer in nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def stamp_sec(header) -> float:
    """Return a ROS timestamp as a floating-point seconds value."""
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def track_id_from_person_id(person_id: str) -> Optional[int]:
    """Extract the raw integer track ID from Block-B's 'person_<id>' string."""
    match = re.fullmatch(r'person_(-?\d+)', str(person_id))
    return None if match is None else int(match.group(1))


@dataclass(frozen=True)
class TrackIdentity:
    """Single tracked person identity rendered onto the frame."""
    track_id: int
    person_id: str
    color_name: str


# Signature of all tracks visible in a single frame, sorted by track_id.
FrameSignature = Tuple[TrackIdentity, ...]


@dataclass
class BufferedScene:
    """One aligned camera scene stored in the rolling buffer."""
    stamp_sec: float
    image: np.ndarray
    observation_header: object
    signature: FrameSignature


@dataclass
class SceneWork:
    """Immutable job passed from ROS callbacks to the VLM inference worker thread."""
    images: List[np.ndarray]
    signature: FrameSignature
    observation_header: object


class QwenLoraBackend:
    """Offline Unsloth inference for a local Qwen2-VL LoRA adapter checkpoint."""

    def __init__(self, adapter_path: Path, base_model: str, load_in_4bit: bool,
                 require_cuda: bool, device_name: str, max_new_tokens: int,
                 min_pixels: int, max_pixels: int, logger) -> None:
        from PIL import Image as PilImageModule
        if not hasattr(PilImageModule, 'Resampling'):
            PilImageModule.Resampling = type('PillowResampling', (), {
                'NEAREST': PilImageModule.NEAREST,
                'LANCZOS': PilImageModule.LANCZOS,
                'BILINEAR': PilImageModule.BILINEAR,
                'BICUBIC': PilImageModule.BICUBIC,
                'BOX': PilImageModule.BOX,
                'HAMMING': PilImageModule.HAMMING,
            })
        import torch
        from unsloth import FastVisionModel

        use_cuda = torch.cuda.is_available()
        if require_cuda and not use_cuda:
            raise RuntimeError('vlm_require_cuda=true but CUDA is unavailable')
        if use_cuda:
            requested_device = torch.device(device_name)
            if requested_device.type != 'cuda':
                raise RuntimeError(
                    f'vlm_device must be CUDA when available, got {device_name!r}')
            device_index = 0 if requested_device.index is None else requested_device.index
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f'vlm_device={device_name!r} invalid; available CUDA count: '
                    f'{torch.cuda.device_count()}')
            self.input_device = torch.device(f'cuda:{device_index}')
            device_map = {'': device_index}
        else:
            self.input_device = torch.device('cpu')
            device_map = None

        with (adapter_path / 'adapter_config.json').open(encoding='utf-8') as config_file:
            adapter_config = json.load(config_file)
        trained_base_model = str(adapter_config.get('base_model_name_or_path', '')).strip()
        if trained_base_model and trained_base_model != base_model:
            logger.warn(
                f'vlm_base_model={base_model!r} differs from adapter base '
                f'{trained_base_model!r}; using adapter base.')

        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=4096,
            dtype=torch.float16 if use_cuda else torch.float32,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            use_gradient_checkpointing=False,
            use_exact_model_name=True,
            fullgraph=False,
            local_files_only=True,
        )
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        self.min_pixels = max(1, min_pixels)
        self.max_pixels = max(1, max_pixels)
        self.torch = torch
        self.max_new_tokens = max(1, max_new_tokens)
        logger.info(
            f'VLM ready via Unsloth: Qwen LoRA={adapter_path}, device={self.input_device}')

    def _state_token_confidences(
        self,
        generated_ids,
        generation_scores,
        decoded_text: str,
    ) -> List[Tuple[str, float]]:
        """Map each generated state string to its geometric-mean token probability."""
        token_ids = generated_ids.tolist()
        if not token_ids or not generation_scores:
            return []

        token_ranges = []
        previous_length = 0
        for index in range(len(token_ids)):
            prefix = self.processor.decode(
                token_ids[:index + 1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            current_length = len(prefix)
            token_ranges.append((previous_length, current_length))
            previous_length = current_length

        state_confidences: List[Tuple[str, float]] = []
        max_scored_index = min(len(token_ids), len(generation_scores))
        for match in STATE_VALUE_PATTERN.finditer(decoded_text):
            state_start, state_end = match.span('state')
            state_token_indices = [
                idx
                for idx, (tok_start, tok_end) in enumerate(token_ranges)
                if idx < max_scored_index and tok_end > state_start and tok_start < state_end
            ]
            if not state_token_indices:
                continue

            token_log_probs = []
            for idx in state_token_indices:
                logits = generation_scores[idx][0].float()
                selected_logit = logits[token_ids[idx]]
                token_log_prob = selected_logit - self.torch.logsumexp(logits, dim=-1)
                token_log_probs.append(float(token_log_prob.item()))

            # Geometric mean prevents longer word pieces from artificially depressing confidence.
            confidence = math.exp(sum(token_log_probs) / len(token_log_probs))
            state_confidences.append((match.group('state').lower(), confidence))

        return state_confidences

    @staticmethod
    def _add_confidences_to_response(
        response: str,
        state_confidences: List[Tuple[str, float]],
    ) -> str:
        """Inject calibrated token confidence into JSON objects in the response text."""
        json_start = response.find('[')
        json_end = response.rfind(']')
        if json_start < 0 or json_end < json_start:
            return response.strip()

        try:
            payload = json.loads(response[json_start:json_end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return response.strip()
        if not isinstance(payload, list):
            return response.strip()

        for item, (scored_state, confidence) in zip(payload, state_confidences):
            if isinstance(item, dict) and str(item.get('state', '')).lower() == scored_state:
                item['confidence'] = round(float(confidence), 4)

        return json.dumps(payload, ensure_ascii=False)

    def infer(self, bgr_images: List[np.ndarray], prompt: str) -> str:
        """Run temporal video inference on 4 oldest-to-newest frames."""
        from PIL import Image as PilImage

        video_frames = [
            PilImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for img in bgr_images
        ]
        if not video_frames:
            raise ValueError('Empty frame list supplied to VLM backend')

        # Package the chronological 4-frame sequence as a video input.
        messages = [{
            'role': 'user',
            'content': [
                {
                    'type': 'video',
                    'video': video_frames,
                },
                {
                    'type': 'text',
                    'text': prompt,
                },
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            videos=[video_frames],
            padding=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            do_sample_frames=False,
            return_tensors='pt',
        )
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        with self.torch.inference_mode():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = generation_output.sequences[:, inputs['input_ids'].shape[1]:]
        decoded_response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        state_confidences = self._state_token_confidences(
            generated_ids[0],
            generation_output.scores,
            decoded_response,
        )
        return self._add_confidences_to_response(decoded_response, state_confidences)


class SocialVlmInteraction(Node):
    """Bridge Block-B RGB-D tracking to offline VLM temporal state classification."""

    def __init__(self) -> None:
        super().__init__('social_vlm_interaction')
        self._declare_parameters()

        self.enabled = bool(self.get_parameter('enable_vlm').value)
        self.gui_enabled = bool(self.get_parameter('enable_vlm_gui').value)
        self.frame_count = max(1, int(self.get_parameter('vlm_frame_count').value))
        self.frame_interval_s = max(0.01, float(self.get_parameter('vlm_frame_interval_s').value))
        self.prompt = str(self.get_parameter('state_prompt').value)
        self.cache_size = max(2, int(self.get_parameter('sync_cache_size').value))

        self.results_dir = Path(str(self.get_parameter('results_dir').value)).expanduser()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.archive_date = ''
        self.archive_index = 0

        # Synchronization caches for Block B inputs.
        self.images: Dict[int, Image] = {}
        self.observations: Dict[int, PeopleObservations] = {}
        self.people: Dict[int, People] = {}
        self.processed_stamps: Deque[int] = deque(maxlen=self.cache_size)

        # 4-frame rolling buffer sampled at fixed sim-time intervals.
        self.frame_buffer: Deque[BufferedScene] = deque(maxlen=self.frame_count)
        self.last_sampled_stamp_sec: Optional[float] = None

        # Concurrency & worker management.
        self.is_busy = False
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.work_queue: queue.Queue[SceneWork] = queue.Queue(maxsize=1)
        self.worker: Optional[threading.Thread] = None
        self.backend: Optional[QwenLoraBackend] = None

        # Visualization state for GUI thread.
        self.ui_snapshot: Optional[List[np.ndarray]] = None
        self.ui_raw_json = ''
        self.ui_latency: Optional[float] = None

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(
            Image, str(self.get_parameter('rgb_topic').value),
            self.image_callback, camera_qos)
        self.create_subscription(
            PeopleObservations, str(self.get_parameter('observations_topic').value),
            self.observations_callback, 10)
        self.create_subscription(
            People, str(self.get_parameter('people_topic').value),
            self.people_callback, 10)

        self.states_pub = self.create_publisher(
            VlmPersonStates, str(self.get_parameter('states_topic').value), 10)
        self.labelled_scene_pub = self.create_publisher(
            Image, str(self.get_parameter('labelled_scene_topic').value), camera_qos)

        if self.enabled:
            self.worker = threading.Thread(
                target=self.vlm_worker,
                name='vlm-inference-worker',
                daemon=True,
            )
            self.worker.start()
        else:
            self.get_logger().info('VLM interaction node is disabled (enable_vlm=false).')

    def _declare_parameters(self) -> None:
        """Declare ROS parameters with sensible defaults."""
        default_prompt = (
            "You are given 4 sequential images from the same short video clip in temporal order "
            "(frame 1 -> frame 4).\n"
            "Each person has a bounding box identified by a color and an ID label.\n"
            "Examine all 4 images together as one temporal sequence.\n\n"
            "Assign exactly one state to each person present in the clip.\n\n"
            "STATE DEFINITIONS\n"
            '- "walking": a person who is moving or translating.\n'
            '- "waiting": a person who remains approximately stationary.\n'
            '- "talking": a person interacting or conversing with another person.\n'
            '- "crossing": moving laterally across the robot forward path.\n'
            '- "approaching": moving toward the robot from the opposite direction.\n'
            '- "straight": moving in the same direction as the robot, ahead of it.\n\n'
            "Determine the state from the movement trajectory across the sequence, not from a single frame.\n"
            'Return only a JSON array with one object per person containing "id" (or "color") and "state".'
        )

        for name, value in {
            'enable_vlm': True,
            'enable_vlm_gui': False,
            'rgb_topic': '/camera/color/image_raw',
            'observations_topic': '/people_observations',
            'people_topic': '/people/tracks',
            'states_topic': '/social_perception/vlm_person_states',
            'labelled_scene_topic': '/social_perception/vlm_input_image',
            'vlm_adapter_path': os.path.expanduser('~/ninorobot2/Saved_Model'),
            'vlm_base_model': 'unsloth/Qwen2-VL-2B-Instruct-bnb-4bit',
            'vlm_load_in_4bit': True,
            'vlm_require_cuda': True,
            'vlm_device': 'cuda:0',
            'vlm_max_new_tokens': 128,
            'vlm_min_pixels': 56 * 56,
            'vlm_max_pixels': 256 * 28 * 28,
            'vlm_frame_count': 4,
            'vlm_frame_interval_s': 0.5,
            'sync_cache_size': 12,
            'state_prompt': default_prompt,
            'results_dir': os.path.expanduser('~/vlm_results'),
        }.items():
            self.declare_parameter(name, value)

    def image_callback(self, message: Image) -> None:
        key = stamp_ns(message.header)
        self._store(self.images, key, message)
        self.try_sample_scene(key)

    def observations_callback(self, message: PeopleObservations) -> None:
        key = stamp_ns(message.header)
        self._store(self.observations, key, message)
        self.try_sample_scene(key)

    def people_callback(self, message: People) -> None:
        key = stamp_ns(message.header)
        self._store(self.people, key, message)
        self.try_sample_scene(key)

    def _store(self, cache: Dict[int, object], key: int, message: object) -> None:
        with self.lock:
            cache[key] = message
            for stale_key in sorted(cache)[:-self.cache_size]:
                del cache[stale_key]

    @staticmethod
    def bgr_image(message: Image) -> np.ndarray:
        if message.encoding not in ('bgr8', 'rgb8'):
            raise ValueError(f'Expected bgr8/rgb8 image, received {message.encoding!r}')
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.width, 3)
        return image if message.encoding == 'bgr8' else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def bgr_message(image: np.ndarray, header: object) -> Image:
        message = Image()
        message.header = header
        message.height, message.width = image.shape[:2]
        message.encoding = 'bgr8'
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = image.tobytes()
        return message

    def try_sample_scene(self, key: int) -> None:
        """Check if all Block-B streams are synchronized for this timestamp."""
        if not self.enabled:
            return

        with self.lock:
            if key in self.processed_stamps:
                return
            image_msg = self.images.get(key)
            observations = self.observations.get(key)
            people_msg = self.people.get(key)
            if image_msg is None or observations is None or people_msg is None:
                return
            self.processed_stamps.append(key)

        try:
            raw_image = self.bgr_image(image_msg)
        except ValueError as err:
            self.get_logger().warn(str(err), throttle_duration_sec=5.0)
            return

        labelled_image, signature = self.render_labels(raw_image, observations, people_msg)
        self.labelled_scene_pub.publish(self.bgr_message(labelled_image, image_msg.header))

        if not signature:
            # No tracked person visible in this frame; do not buffer empty scenes.
            return

        self.update_rolling_buffer(image_msg.header, labelled_image, signature)

    @staticmethod
    def render_labels(
        image: np.ndarray,
        observations: PeopleObservations,
        people_msg: People,
    ) -> Tuple[np.ndarray, FrameSignature]:
        """Render distinct bounding box colors and 'ID <id>' labels for each person.

        This guarantees visual grounding whether the model predicts by ID or by color.
        """
        labelled = image.copy()
        height, width = labelled.shape[:2]
        confirmed_ids: Set[str] = {p.id for p in people_msg.people}
        identities: List[TrackIdentity] = []

        for obs in observations.people:
            if obs.id not in confirmed_ids:
                continue
            track_id = track_id_from_person_id(obs.id)
            if track_id is None:
                continue

            left = max(0, min(width - 1, int(obs.bbox_x_min)))
            top = max(0, min(height - 1, int(obs.bbox_y_min)))
            right = max(0, min(width - 1, int(obs.bbox_x_max)))
            bottom = max(0, min(height - 1, int(obs.bbox_y_max)))
            if right <= left or bottom <= top:
                continue

            # Deterministic color assignment based on track_id.
            color_index = track_id % len(PALETTE_BGR)
            bgr_color = PALETTE_BGR[color_index]
            color_name = COLOR_NAMES[color_index]

            cv2.rectangle(labelled, (left, top), (right, bottom), bgr_color, 2)
            cv2.putText(
                labelled,
                f'ID {track_id} ({color_name})',
                (left, max(20, top - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                bgr_color,
                2,
            )
            identities.append(TrackIdentity(track_id, obs.id, color_name))

        identities.sort(key=lambda item: item.track_id)
        return labelled, tuple(identities)

    def update_rolling_buffer(
        self,
        header: object,
        labelled_image: np.ndarray,
        signature: FrameSignature,
    ) -> None:
        """Append to rolling buffer strictly sampled every 0.5s sim-time."""
        current_stamp_sec = stamp_sec(header)

        with self.lock:
            # Handle simulation clock resets.
            if (self.last_sampled_stamp_sec is not None and
                    current_stamp_sec < self.last_sampled_stamp_sec):
                self.get_logger().warn('Clock reset detected in header timestamp; clearing buffer.')
                self.frame_buffer.clear()
                self.last_sampled_stamp_sec = None

            # Sample strictly at the configured temporal interval.
            if (self.last_sampled_stamp_sec is not None and
                    (current_stamp_sec - self.last_sampled_stamp_sec) < self.frame_interval_s):
                return

            self.last_sampled_stamp_sec = current_stamp_sec
            self.frame_buffer.append(BufferedScene(
                stamp_sec=current_stamp_sec,
                image=labelled_image.copy(),
                observation_header=deepcopy(header),
                signature=signature,
            ))

            if len(self.frame_buffer) < self.frame_count:
                return

            # Check track signature continuity across all 4 frames.
            all_signatures = [entry.signature for entry in self.frame_buffer]
            is_clean = bool(all_signatures[0]) and all(
                s == all_signatures[0] for s in all_signatures[1:]
            )

            if not is_clean:
                self.get_logger().warn(
                    f'Frame buffer track signature changed across 4-frame window; skipping.',
                    throttle_duration_sec=2.0,
                )
                return

            # If worker is idle, dispatch the 4-frame job.
            if not self.is_busy:
                self.is_busy = True
                job = SceneWork(
                    images=[entry.image for entry in self.frame_buffer],
                    signature=all_signatures[-1],
                    observation_header=deepcopy(self.frame_buffer[-1].observation_header),
                )
                try:
                    self.work_queue.put_nowait(job)
                except queue.Full:
                    self.is_busy = False

    def vlm_worker(self) -> None:
        """Worker thread executing Qwen inference asynchronously."""
        adapter_path = Path(str(self.get_parameter('vlm_adapter_path').value)).expanduser()
        if not adapter_path.is_dir() or not (adapter_path / 'adapter_config.json').is_file():
            self.get_logger().error(f'LoRA adapter not found at: {adapter_path}')
            return

        try:
            self.backend = QwenLoraBackend(
                adapter_path=adapter_path,
                base_model=str(self.get_parameter('vlm_base_model').value),
                load_in_4bit=bool(self.get_parameter('vlm_load_in_4bit').value),
                require_cuda=bool(self.get_parameter('vlm_require_cuda').value),
                device_name=str(self.get_parameter('vlm_device').value),
                max_new_tokens=int(self.get_parameter('vlm_max_new_tokens').value),
                min_pixels=int(self.get_parameter('vlm_min_pixels').value),
                max_pixels=int(self.get_parameter('vlm_max_pixels').value),
                logger=self.get_logger(),
            )
        except Exception as err:
            self.get_logger().error(f'Failed to initialize QwenLoraBackend: {err}')
            return

        while not self.stop_event.is_set():
            try:
                job = self.work_queue.get(timeout=0.25)
            except queue.Empty:
                continue

            t_start = time.monotonic()
            try:
                raw_response = self.backend.infer(job.images, self.prompt)
            except Exception as err:
                raw_response = f'Inference Error: {err}'
                self.get_logger().error(f'VLM inference exception: {err}')

            latency = time.monotonic() - t_start
            self.publish_state_result(job, raw_response)

            self.get_logger().info(f'VLM latency = {latency:.2f} s')
            if self.gui_enabled:
                with self.lock:
                    self.ui_snapshot = job.images
                    self.ui_raw_json = raw_response
                    self.ui_latency = latency

            try:
                self.save_result_visualization(job.images, raw_response, latency)
            except Exception as err:
                self.get_logger().warn(f'Failed to save VLM result visualization: {err}')
            finally:
                with self.lock:
                    self.is_busy = False

    @staticmethod
    def _parse_json_array(response: str):
        start = response.find('[')
        end = response.rfind(']')
        if start < 0 or end < start:
            return []
        try:
            payload = json.loads(response[start:end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    def publish_state_result(self, job: SceneWork, raw_response: str) -> None:
        """Parse predictions and map back to stable ByteTrack IDs."""
        id_map: Dict[int, TrackIdentity] = {item.track_id: item for item in job.signature}
        color_map: Dict[str, TrackIdentity] = {item.color_name.lower(): item for item in job.signature}

        output = VlmPersonStates()
        output.header = job.observation_header
        output.inference_stamp = self.get_clock().now().to_msg()
        output.raw_response = raw_response

        resolved_tracks: Set[int] = set()

        for item in self._parse_json_array(raw_response):
            if not isinstance(item, dict):
                continue
            state = str(item.get('state', '')).strip().lower()
            if not state:
                continue

            identity: Optional[TrackIdentity] = None
            # Support both numeric ID and color name resolutions.
            if 'id' in item:
                try:
                    parsed_id = int(item['id'])
                    identity = id_map.get(parsed_id)
                except (TypeError, ValueError):
                    pass

            if identity is None and 'color' in item:
                parsed_color = str(item['color']).strip().lower()
                identity = color_map.get(parsed_color)

            if identity is None or identity.track_id in resolved_tracks:
                continue

            person_state = VlmPersonState()
            person_state.track_id = int(identity.track_id)
            person_state.person_id = identity.person_id
            person_state.state = state
            output.states.append(person_state)
            resolved_tracks.add(identity.track_id)

        # Fallback to unknown for tracked persons not mentioned in the model output.
        for identity in job.signature:
            if identity.track_id in resolved_tracks:
                continue
            person_state = VlmPersonState()
            person_state.track_id = int(identity.track_id)
            person_state.person_id = identity.person_id
            person_state.state = 'unknown'
            output.states.append(person_state)

        self.states_pub.publish(output)
        self.get_logger().info(
            f'Published VLM states: {len(resolved_tracks)}/{len(job.signature)} resolved '
            f'from frame {output.header.stamp.sec}.{output.header.stamp.nanosec:09d}')

    def compose_visualization(
        self,
        images: List[np.ndarray],
        raw_json: str,
        latency: Optional[float],
    ) -> np.ndarray:
        """Compose a 2x2 grid with an informative bottom diagnostic banner."""
        target_w, target_h = 400, 225
        resized_imgs = []

        for i, img in enumerate(images):
            resized = cv2.resize(img, (target_w, target_h))
            label = f'Frame {i + 1} (t={i * self.frame_interval_s:.1f}s)'
            cv2.rectangle(resized, (5, 5), (170, 30), (0, 0, 0), -1)
            cv2.putText(resized, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            resized_imgs.append(resized)

        top_row = np.hstack([resized_imgs[0], resized_imgs[1]])
        bottom_row = np.hstack([resized_imgs[2], resized_imgs[3]])
        grid = np.vstack([top_row, bottom_row])

        banner_h = 190
        banner = np.zeros((banner_h, grid.shape[1], 3), dtype=np.uint8)

        lat_text = f'VLM Latency: {latency:.2f} s' if latency is not None else 'VLM Latency: --'
        cv2.putText(banner, lat_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(banner, 'RAW JSON OUTPUT (+ token confidence):', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        clean_json = raw_json.replace('\n', ' ')
        max_chars = 95
        lines = [clean_json[i:i + max_chars] for i in range(0, len(clean_json), max_chars)]
        y_offset = 88
        for line in lines[:5]:
            cv2.putText(banner, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20

        return np.vstack([grid, banner])

    def render_visualization(
        self,
        images: List[np.ndarray],
        raw_json: str,
        latency: Optional[float],
    ) -> None:
        combined = self.compose_visualization(images, raw_json, latency)
        cv2.imshow('VLM Interaction 4-Frame Window', combined)
        cv2.waitKey(1)

    def next_result_path(self) -> Path:
        date_code = datetime.now().strftime('%m%d')
        if date_code != self.archive_date:
            existing = []
            for path in self.results_dir.glob(f'vlm_{date_code}_*.png'):
                suffix = path.stem.rsplit('_', 1)[-1]
                if suffix.isdigit():
                    existing.append(int(suffix))
            self.archive_date = date_code
            self.archive_index = max(existing, default=0)

        self.archive_index += 1
        return self.results_dir / f'vlm_{date_code}_{self.archive_index:04d}.png'

    def save_result_visualization(
        self,
        images: List[np.ndarray],
        raw_json: str,
        latency: float,
    ) -> None:
        output_path = self.next_result_path()
        combined = self.compose_visualization(images, raw_json, latency)
        cv2.imwrite(str(output_path), combined)

    def destroy_node(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
        if self.gui_enabled:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SocialVlmInteraction()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    ros_thread: Optional[threading.Thread] = None
    try:
        if not node.gui_enabled:
            executor.spin()
        else:
            ros_thread = threading.Thread(
                target=executor.spin,
                name='ros-executor',
                daemon=True,
            )
            ros_thread.start()

            while rclpy.ok() and not node.stop_event.is_set():
                with node.lock:
                    snapshot = node.ui_snapshot
                    raw_json = node.ui_raw_json
                    latency = node.ui_latency

                if snapshot is not None and len(snapshot) == node.frame_count:
                    node.render_visualization(snapshot, raw_json, latency)

                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC key
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        executor.shutdown()
        node.destroy_node()
        if ros_thread is not None:
            ros_thread.join(timeout=1.0)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
