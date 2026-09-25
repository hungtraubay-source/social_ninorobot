#!/usr/bin/env python3
"""Classify per-person temporal social states with a 4-frame rolling buffer.

This node connects Block-B perception with the fine-tuned Qwen2-VL LoRA model.
It subscribes to Block B's synchronized output topics:
  - ``sensor_msgs/Image``: raw camera frames.
  - ``social_perception/PeopleObservations``: 2D bounding boxes with ByteTrack IDs.
  - ``social_perception/People``: validated 3D map/odom localized positions.

Workflow & Design Decisions:
1. Grounding & Identity Binding:
   Each detected track is assigned a distinct bounding-box color (e.g. "green",
   "blue") and a label 'ID <id> (<color>)' rendered on the frame.  The VLM model
   returns {"color": "<name>", "state": "<state>"} — the color is the primary
   grounding key that maps back to a stable ByteTrack track_id / person_id.
2. Sim-Time Rolling Buffer:
   Maintains a 4-frame rolling buffer strictly sampled at fixed intervals (default
   0.5 s) using ROS simulation time (header stamp). Handles clock resets.
3. Signature Continuity:
   Snapshots are enqueued only when track identities are consistent across all 4
   frames and the inference worker is idle.
4. Video-Structure Qwen Inference & Token Confidence:
   Frames are packaged as a single temporal video sequence for Qwen2-VL.
   Geometric-mean token probability is computed from output logits and stored in
   the raw JSON response for diagnostics.
5. Deterministic Safety Contract:
   Output topic ``/social_perception/vlm_person_states`` carries semantic labels only
   (e.g., 'talking', 'crossing').  Position, velocity, and costmap geometry remain
   strictly owned by deterministic upstream modules.  semantic_fusion.py joins these
   labels to live People tracks via the person_id field.
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

# Palette of distinct BGR colors for multi-person bounding-box rendering.
# Index i maps to COLOR_NAMES[i]; the model receives color names as grounding keys.
PALETTE_BGR = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]

COLOR_NAMES = ['green', 'blue', 'red', 'yellow', 'magenta']

# Regex that locates the value of a "state" key inside a JSON object so we can
# calculate per-state token confidence from model logits.
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
    person_id: str  # stable Block-B identifier, e.g. "person_1"
    color_name: str  # name used as grounding key in the rendered label


# Ordered tuple of identities present in one buffer frame, sorted by track_id.
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
        # Pillow < 9.1 lacks the Resampling enum; patch it so Unsloth works.
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
        self.frame_count = max(1, int(self.get_parameter('vlm_frame_count').value))
        self.frame_interval_s = max(0.01, float(self.get_parameter('vlm_frame_interval_s').value))
        self.prompt = str(self.get_parameter('state_prompt').value)
        self.cache_size = max(2, int(self.get_parameter('sync_cache_size').value))

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
        # Publish labelled frames for debugging (rqt_image_view or RViz).
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
        """Declare ROS parameters with sensible defaults.

        The default prompt instructs the model to return {"color": ..., "state": ...}
        so that publish_state_result can directly look up the color in the palette map
        and recover track_id / person_id without ambiguity.
        """
        default_prompt = (
            'You are given 4 sequential images from the same short video clip in temporal order '
            '(frame 1 -> frame 4).\n'
            'Each person has a bounding box labeled "ID <number> (<color>)".\n'
            'Examine all 4 images together as one temporal sequence.\n\n'
            'Assign exactly one state to each person.\n\n'
            'STATE DEFINITIONS\n'
            '- "crossing": moving laterally across the robot forward path.\n'
            '- "approaching": moving toward the robot from the opposite direction.\n'
            '- "straight": moving in the same direction as the robot, ahead of it.\n'
            '- "talking": interacting or conversing with another person.\n\n'
            'Determine the state from movement trajectory across the sequence, not from a single frame.\n'
            'Return ONLY a JSON array. Each object must have:\n'
            '  "ID": the integer number from the bounding box label (e.g. 1, 2),\n'
            '  "state": one of the states above.\n'
            'Example: [{"ID": 1, "state": "talking"}, {"ID": 2, "state": "crossing"}]'
        )

        for name, value in {
            'enable_vlm': True,
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
            'vlm_max_new_tokens': 64,
            'vlm_min_pixels': 56 * 56,
            'vlm_max_pixels': 128 * 28 * 28,
            'vlm_frame_count': 4,
            'vlm_frame_interval_s': 0.5,
            'sync_cache_size': 12,
            'state_prompt': default_prompt,
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

    def render_labels(
        self,
        image: np.ndarray,
        observations: PeopleObservations,
        people_msg: People,
    ) -> Tuple[np.ndarray, FrameSignature]:
        """Render color-coded bounding boxes with 'ID <id> (<color>)' labels.

        Colors are assigned by order of first appearance via ``_track_color_map``
        so two simultaneously visible tracks always receive distinct colors,
        regardless of their numeric track_id values.
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
        """Append to rolling buffer strictly sampled every frame_interval_s sim-time."""
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

            # Check track signature continuity across the 4-frame window.
            all_signatures = [entry.signature for entry in self.frame_buffer]
            newest_signature = all_signatures[-1]
            newest_ids = {item.track_id for item in newest_signature}

            # Require that the newest frame has tracks and at least one track
            # was present in earlier frames so the model has temporal context.
            # This tolerates single-frame detector flicker while still ensuring
            # the scene is coherent.
            has_continuity = bool(newest_ids) and any(
                bool(newest_ids & {item.track_id for item in s})
                for s in all_signatures[:-1]
            )

            if not has_continuity:
                self.get_logger().warn(
                    f'Frame buffer lacks track continuity for {sorted(newest_ids)}; skipping.',
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

            with self.lock:
                self.is_busy = False

    @staticmethod
    def _parse_json_array(response: str) -> list:
        """Extract the first JSON array from a model response string."""
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
        """Parse color+state pairs from the model and map to stable ByteTrack identities.

        The fine-tuned Qwen2-VL model returns objects like {"color": "green", "state": "talking"}.
        The color is the grounding key rendered on every bounding box label.  This method
        looks up the color in the palette map to recover track_id and person_id, then
        publishes VlmPersonStates so that semantic_fusion.py can join them via person_id.

        Fallback: any tracked person whose color is absent in the model response is published
        with state="unknown", which semantic_fusion ignores.
        """
        # Build a color -> identity lookup from the job's rendered palette.
        color_map: Dict[str, TrackIdentity] = {
            item.color_name.lower(): item for item in job.signature
        }
        # Also support numeric ID as a secondary fallback in case the model uses it.
        id_map: Dict[int, TrackIdentity] = {item.track_id: item for item in job.signature}

        output = VlmPersonStates()
        output.header = job.observation_header
        output.inference_stamp = self.get_clock().now().to_msg()
        output.raw_response = raw_response

        self.get_logger().info(
            f'VLM raw: {raw_response!r} | '
            f'ids={list(id_map.keys())} colors={list(color_map.keys())}')

        resolved_tracks: Set[int] = set()

        for item in self._parse_json_array(raw_response):
            if not isinstance(item, dict):
                continue
            state = str(item.get('state', '')).strip().lower()
            if not state:
                continue

            identity: Optional[TrackIdentity] = None

            # Primary route: model returns {"ID": 1, "state": ...} — numeric int.
            # Check both "ID" and "id" keys (case-insensitive).
            id_val = item.get('ID', item.get('id'))
            if id_val is not None:
                try:
                    identity = id_map.get(int(id_val))
                except (TypeError, ValueError):
                    # Model put a string (e.g. color name) in the ID field — try color map.
                    identity = color_map.get(str(id_val).strip().lower())

            # Fallback: model returns {"color": "green", "state": ...}.
            if identity is None and 'color' in item:
                identity = color_map.get(str(item['color']).strip().lower())

            if identity is None or identity.track_id in resolved_tracks:
                continue

            person_state = VlmPersonState()
            person_state.track_id = int(identity.track_id)
            person_state.person_id = identity.person_id  # used by semantic_fusion.py
            person_state.color = identity.color_name      # raw model grounding key
            person_state.state = state
            output.states.append(person_state)
            resolved_tracks.add(identity.track_id)

        # Publish unknown for any tracked person not mentioned in the model output.
        for identity in job.signature:
            if identity.track_id in resolved_tracks:
                continue
            person_state = VlmPersonState()
            person_state.track_id = int(identity.track_id)
            person_state.person_id = identity.person_id
            person_state.color = identity.color_name
            person_state.state = 'unknown'
            output.states.append(person_state)

        self.states_pub.publish(output)
        self.get_logger().info(
            f'Published VLM states: {len(resolved_tracks)}/{len(job.signature)} resolved '
            f'from frame {output.header.stamp.sec}.{output.header.stamp.nanosec:09d}')

    def destroy_node(self):
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SocialVlmInteraction()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_event.set()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
