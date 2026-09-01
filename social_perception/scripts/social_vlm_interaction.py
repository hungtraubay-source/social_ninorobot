#!/usr/bin/env python3
"""Classify per-person temporal states without delaying RGB-D tracking.

Input topics from Block B share the same RGB header timestamp:
``sensor_msgs/Image`` supplies the full camera frame, ``PeopleObservations``
supplies ByteTrack IDs and 2-D boxes, and ``People`` confirms each visible ID
has a map-frame pose. This node renders ``ID <track_id>`` on four chronological
full-scene images and sends those images to the optional Qwen LoRA worker.

Output ``/social_perception/vlm_person_states`` uses ``VlmPersonStates``. It
contains only semantic labels such as ``crossing``. RGB-D, ByteTrack, TF, and
all geometry/costmap decisions remain owned by deterministic pipeline blocks.
"""

import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

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


def stamp_ns(header) -> int:
    """Return one ROS timestamp as an exact cache key in nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def track_id_from_person_id(person_id: str) -> Optional[int]:
    """Extract the raw ByteTrack integer from Block-B's ``person_<id>`` key."""
    match = re.fullmatch(r'person_(-?\d+)', str(person_id))
    return None if match is None else int(match.group(1))


def parse_person_states(response: str,
                        visible_track_ids: Dict[int, str]) -> Tuple[List[Tuple[int, str]], str]:
    """Validate the fine-tune JSON array against IDs visible in frame four.

    The model is allowed to return a JSON array only. An unknown ID is rejected
    rather than attached to a newly recycled ByteTrack identity. Empty arrays
    are valid and represent no labelled visible person in this scene.
    """
    start = response.find('[')
    if start < 0:
        return [], 'model response does not contain a JSON array'
    try:
        payload, _ = json.JSONDecoder().raw_decode(response[start:])
    except json.JSONDecodeError as error:
        return [], f'invalid JSON array: {error.msg}'
    if not isinstance(payload, list):
        return [], 'model JSON root is not an array'

    states: List[Tuple[int, str]] = []
    seen_ids = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        model_id = item.get('id')
        if isinstance(model_id, bool):
            continue
        try:
            track_id = int(model_id)
        except (TypeError, ValueError):
            continue
        state = str(item.get('state', '')).strip()
        if not state or track_id not in visible_track_ids or track_id in seen_ids:
            continue
        seen_ids.add(track_id)
        states.append((track_id, state))
    return states, ''


class QwenLoraBackend:
    """Offline Unsloth inference for one local Qwen2-VL LoRA adapter.

    ``Saved_Model/adapter_config.json`` was trained against an Unsloth
    pre-quantized ``*-bnb-4bit`` base.  It must therefore be opened through
    ``FastVisionModel`` so Unsloth restores bitsandbytes' serialized FP4/NF4
    metadata before the visual merger receives an image.
    """

    def __init__(self, adapter_path: Path, base_model: str, load_in_4bit: bool,
                 require_cuda: bool, device_name: str, max_new_tokens: int,
                 min_pixels: int, max_pixels: int, logger) -> None:
        # Import lazily so RGB-D tracking stays available when this optional
        # semantic worker is disabled or its GPU dependencies are unavailable.
        from PIL import Image as PilImageModule
        if not hasattr(PilImageModule, 'Resampling'):
            # Transformers releases used on Ubuntu 22.04 expect the newer
            # Pillow enum, while Pillow 9 exposes equivalent legacy constants.
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
            # The complete 4-bit vision-language model must remain on one GPU.
            # A device map with CPU offload is invalid for this pre-quantized
            # checkpoint because its visual merger consumes FP4/NF4 weights.
            requested_device = torch.device(device_name)
            if requested_device.type != 'cuda':
                raise RuntimeError(
                    f'vlm_device must select CUDA when CUDA is available, got {device_name!r}')
            device_index = 0 if requested_device.index is None else requested_device.index
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f'vlm_device={device_name!r} is unavailable; CUDA device count is '
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
            # The adapter was trained for this exact base; choosing a different
            # ROS parameter would silently produce incompatible LoRA weights.
            logger.warn(
                f'vlm_base_model={base_model!r} differs from the adapter base '
                f'{trained_base_model!r}; the adapter base is used.')

        # Pass the adapter directory, not the base name.  FastVisionModel reads
        # adapter_config.json, resolves its trained base, restores the pre-
        # quantized BnB tensors, then applies adapter_model.safetensors.  The
        # generic Transformers/PEFT route loses quant_state in visual.merger.
        self.model, self.processor = FastVisionModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=4096,
            dtype=torch.float16 if use_cuda else torch.float32,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
            use_gradient_checkpointing=False,
            use_exact_model_name=True,
            fullgraph=False,
            # The robot is offline: cache miss must fail rather than download.
            local_files_only=True,
        )
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        # Qwen2-VL accepts these as call-time processor arguments.  Do not
        # mutate ``processor.image_processor``: Unsloth's processor exposes
        # read-only compatibility properties on current Transformers releases.
        self.min_pixels = max(1, min_pixels)
        self.max_pixels = max(1, max_pixels)
        self.torch = torch
        self.max_new_tokens = max(1, max_new_tokens)
        logger.info(
            f'VLM ready through Unsloth: Qwen LoRA={adapter_path}, device={self.input_device}, '
            f'device_map={getattr(self.model, "hf_device_map", device_map)}')

    def infer(self, bgr_images: List[np.ndarray], prompt: str) -> str:
        """Run four oldest-to-newest labelled scene frames as Qwen input."""
        from PIL import Image as PilImage

        images = [PilImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                  for image in bgr_images]
        if not images:
            raise ValueError('VLM sequence has no images')
        messages = [{'role': 'user', 'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # The limits bound visual-token memory for all four RGB frames.  Qwen
        # applies them during its image resize step before the visual merger.
        inputs = self.processor(text=[text], images=images, padding=False,
                                min_pixels=self.min_pixels,
                                max_pixels=self.max_pixels,
                                return_tensors='pt')
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()


@dataclass(frozen=True)
class SceneFrame:
    """One labelled camera scene sampled at a Block-B RGB timestamp."""

    stamp_ns: int
    image: np.ndarray
    observation_header: object
    visible_track_ids: Dict[int, str]


@dataclass(frozen=True)
class SceneWork:
    """Immutable four-frame VLM request passed from ROS callbacks to worker."""

    images: List[np.ndarray]
    observation_header: object
    visible_track_ids: Dict[int, str]


class SocialVlmInteraction(Node):
    """Independent state worker: Block-B scene frames -> per-track VLM states."""

    def __init__(self) -> None:
        super().__init__('social_vlm_interaction')
        self._declare_parameters()
        self.enabled = bool(self.get_parameter('enable_vlm').value)
        self.interval_s = max(0.0, float(self.get_parameter('inference_interval_s').value))
        self.frame_count = max(1, int(self.get_parameter('vlm_frame_count').value))
        self.frame_interval_s = max(0.0, float(
            self.get_parameter('vlm_frame_interval_s').value))
        self.prompt = str(self.get_parameter('state_prompt').value)
        self.cache_size = max(2, int(self.get_parameter('sync_cache_size').value))
        self.images: Dict[int, Image] = {}
        self.observations: Dict[int, PeopleObservations] = {}
        self.people: Dict[int, People] = {}
        # This rolling buffer keeps full scenes. A four-frame request spans
        # 1.5 s with the default 0.5 s sampling interval.
        self.scene_frames: Deque[SceneFrame] = deque(maxlen=self.frame_count)
        self.processed_scene_stamps: Deque[int] = deque(maxlen=self.cache_size)
        self.last_inference_time = float('-inf')
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.work_queue: queue.Queue[SceneWork] = queue.Queue(maxsize=1)
        self.worker: Optional[threading.Thread] = None
        self.backend: Optional[QwenLoraBackend] = None

        camera_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE)
        # All callbacks cache by the common RGB header. The scene is sampled
        # only after pixels, 2-D IDs, and map-localized tracks all agree.
        self.create_subscription(Image, str(self.get_parameter('rgb_topic').value),
                                 self.image_callback, camera_qos)
        self.create_subscription(PeopleObservations,
                                 str(self.get_parameter('observations_topic').value),
                                 self.observations_callback, 10)
        self.create_subscription(People, str(self.get_parameter('people_topic').value),
                                 self.people_callback, 10)
        self.states_pub = self.create_publisher(
            VlmPersonStates, str(self.get_parameter('states_topic').value), 10)
        self.labelled_scene_pub = self.create_publisher(
            Image, str(self.get_parameter('labelled_scene_topic').value), camera_qos)

        if self.enabled:
            self.worker = threading.Thread(target=self.vlm_worker,
                                           name='social-vlm-state', daemon=True)
            self.worker.start()
        else:
            self.get_logger().info('VLM state worker disabled; no model is loaded.')

    def _declare_parameters(self) -> None:
        """Declare all ROS I/O, model, timing, and prompt parameters."""
        for name, value in {
            'enable_vlm': False,
            # Input: Block-B topics with matching RGB timestamps.
            'rgb_topic': '/camera/color/image_raw',
            'observations_topic': '/people_observations',
            'people_topic': '/people/tracks',
            # Output: semantic-only state; Nav2 has no subscriber in this step.
            'states_topic': '/social_perception/vlm_person_states',
            # Debug output: newest RGB scene exactly as labelled for Qwen.
            'labelled_scene_topic': '/social_perception/vlm_input_image',
            'vlm_adapter_path': 'Saved_Model',
            'vlm_base_model': 'unsloth/Qwen2-VL-2B-Instruct-bnb-4bit',
            'vlm_load_in_4bit': True,
            'vlm_require_cuda': True,
            # Keep all 4-bit layers on this GPU; do not let Accelerate auto-offload FP4.
            'vlm_device': 'cuda:0',
            # A JSON array can be longer than the old talking/not-talking reply.
            'vlm_max_new_tokens': 64,
            'vlm_min_pixels': 56 * 56,
            'vlm_max_pixels': 256 * 28 * 28,
            # Four samples with this spacing cover 1.5 s from first to fourth.
            'vlm_frame_count': 4,
            'vlm_frame_interval_s': 0.5,
            # Limit VLM load while keeping the rolling scene history current.
            'inference_interval_s': 2.0,
            'sync_cache_size': 12,
            'state_prompt': (
                'You are given 4 sequential images from the same short video clip in '
                'temporal order (frame 1 is the oldest and frame 4 is the newest), '
                'not from a single frame. Return a JSON array with one object '
                'containing "id" and "state" for each person. Return only valid JSON.'),
        }.items():
            self.declare_parameter(name, value)

    def image_callback(self, message: Image) -> None:
        """Cache RGB pixels and sample if Block-B metadata arrived first."""
        key = stamp_ns(message.header)
        self._store(self.images, key, message)
        self.try_enqueue(key)

    def observations_callback(self, message: PeopleObservations) -> None:
        """Cache image-space ByteTrack boxes used to label VLM input images."""
        key = stamp_ns(message.header)
        self._store(self.observations, key, message)
        self.try_enqueue(key)

    def people_callback(self, message: People) -> None:
        """Cache map-frame tracks and sample if RGB plus boxes are available."""
        key = stamp_ns(message.header)
        self._store(self.people, key, message)
        self.try_enqueue(key)

    def _store(self, cache: Dict[int, object], key: int, message: object) -> None:
        """Retain a short aligned buffer and bound memory at camera frame rate."""
        with self.lock:
            cache[key] = message
            for stale_key in sorted(cache)[:-self.cache_size]:
                del cache[stale_key]

    @staticmethod
    def bgr_image(message: Image) -> np.ndarray:
        """Convert supported ROS RGB encodings to contiguous OpenCV BGR pixels."""
        if message.encoding not in ('bgr8', 'rgb8'):
            raise ValueError(f'expected bgr8/rgb8 image, received {message.encoding!r}')
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.width, 3)
        return image if message.encoding == 'bgr8' else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def bgr_message(image: np.ndarray, header: object) -> Image:
        """Wrap a labelled OpenCV BGR scene for the VLM input debug topic."""
        message = Image()
        message.header = header
        message.height, message.width = image.shape[:2]
        message.encoding = 'bgr8'
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = image.tobytes()
        return message

    def try_enqueue(self, key: int) -> None:
        """Sample one labelled full scene after all Block-B data shares one stamp."""
        if not self.enabled:
            return
        with self.lock:
            if key in self.processed_scene_stamps:
                return
            image_msg = self.images.get(key)
            observations = self.observations.get(key)
            people_message = self.people.get(key)
            if image_msg is None or observations is None or people_message is None:
                return
            self.processed_scene_stamps.append(key)
        try:
            image = self.bgr_image(image_msg)
        except ValueError as error:
            self.get_logger().warn(str(error), throttle_duration_sec=5.0)
            return
        labelled_image, visible_track_ids = self.labelled_scene(
            image, observations, people_message)
        # Publish this exact frame so a tester can confirm the model receives
        # the expected ByteTrack IDs, boxes, and image coordinate convention.
        self.labelled_scene_pub.publish(self.bgr_message(labelled_image, image_msg.header))
        if not visible_track_ids:
            # A model output cannot be mapped safely when no current ByteTrack
            # ID is visible on the frame sent to Qwen.
            return
        self.collect_scene(key, labelled_image, people_message.header, visible_track_ids)

    @staticmethod
    def labelled_scene(image: np.ndarray, observations: PeopleObservations,
                       people_message: People) -> Tuple[np.ndarray, Dict[int, str]]:
        """Render current ByteTrack IDs on the whole RGB frame for Qwen.

        The model returns a numeric ``id``. Rendering the same numeric label on
        every input frame is the explicit visual binding from model output to
        Block-B tracking identity; raw camera pixels alone cannot provide it.
        """
        labelled = image.copy()
        height, width = labelled.shape[:2]
        person_ids = {person.id for person in people_message.people}
        visible_track_ids: Dict[int, str] = {}
        for observation in observations.people:
            if observation.id not in person_ids:
                continue
            track_id = track_id_from_person_id(observation.id)
            if track_id is None:
                continue
            left = max(0, min(width - 1, int(observation.bbox_x_min)))
            top = max(0, min(height - 1, int(observation.bbox_y_min)))
            right = max(0, min(width - 1, int(observation.bbox_x_max)))
            bottom = max(0, min(height - 1, int(observation.bbox_y_max)))
            if right <= left or bottom <= top:
                continue
            # Yellow is reserved for VLM input labels, separate from the red
            # invalid-depth overlay in Block B's operator-facing image.
            color = (0, 255, 255)
            cv2.rectangle(labelled, (left, top), (right, bottom), color, 2)
            cv2.putText(labelled, f'ID {track_id}', (left, max(20, top - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            visible_track_ids[track_id] = observation.id
        return labelled, visible_track_ids

    def collect_scene(self, current_stamp_ns: int, labelled_image: np.ndarray,
                      observation_header: object,
                      visible_track_ids: Dict[int, str]) -> None:
        """Maintain a rolling chronological scene window and enqueue it for Qwen."""
        if self.scene_frames:
            elapsed_s = (current_stamp_ns - self.scene_frames[-1].stamp_ns) / 1_000_000_000.0
            if elapsed_s < self.frame_interval_s:
                return
        self.scene_frames.append(SceneFrame(
            current_stamp_ns, labelled_image, observation_header, visible_track_ids))
        if len(self.scene_frames) < self.frame_count:
            return
        now = time.monotonic()
        if now - self.last_inference_time < self.interval_s:
            return
        latest = self.scene_frames[-1]
        work = SceneWork([frame.image for frame in self.scene_frames],
                         latest.observation_header, latest.visible_track_ids)
        self.last_inference_time = now
        # Latest-only prevents a slow VLM response from deciding about a scene
        # which ByteTrack has already replaced with newer observations.
        try:
            self.work_queue.put_nowait(work)
        except queue.Full:
            try:
                self.work_queue.get_nowait()
            except queue.Empty:
                pass
            self.work_queue.put_nowait(work)

    def vlm_worker(self) -> None:
        """Load Qwen once, then classify scene windows outside ROS callbacks."""
        adapter = Path(str(self.get_parameter('vlm_adapter_path').value)).expanduser()
        if not adapter.is_dir() or not (adapter / 'adapter_config.json').is_file():
            self.get_logger().error(
                f'VLM adapter not found or not PEFT LoRA: {adapter}; state output is disabled.')
            return
        try:
            self.backend = QwenLoraBackend(
                adapter, str(self.get_parameter('vlm_base_model').value),
                bool(self.get_parameter('vlm_load_in_4bit').value),
                bool(self.get_parameter('vlm_require_cuda').value),
                str(self.get_parameter('vlm_device').value),
                int(self.get_parameter('vlm_max_new_tokens').value),
                int(self.get_parameter('vlm_min_pixels').value),
                int(self.get_parameter('vlm_max_pixels').value), self.get_logger())
        except Exception as error:
            self.get_logger().error(f'Cannot load local VLM: {type(error).__name__}: {error!r}')
            return
        while not self.stop_event.is_set():
            try:
                work = self.work_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                response = self.backend.infer(work.images, self.prompt)
                states, parse_error = parse_person_states(response, work.visible_track_ids)
            except Exception as error:
                response = f'{type(error).__name__}: {error!r}'
                states, parse_error = [], response
                self.get_logger().error(f'VLM scene inference failed: {response}')
            self.publish_result(work, states, response)
            if parse_error:
                self.get_logger().warn(
                    f'VLM response was not accepted: {parse_error}; raw={response!r}')
            self.get_logger().info(
                f'VLM scene: frames={len(work.images)}, states={states}, '
                f'latency={time.monotonic() - started:.2f}s')

    def publish_result(self, work: SceneWork, states: List[Tuple[int, str]],
                       response: str) -> None:
        """Publish typed semantic states tied to frame-four ByteTrack IDs."""
        output = VlmPersonStates()
        output.header = work.observation_header
        output.inference_stamp = self.get_clock().now().to_msg()
        output.raw_response = response
        for track_id, state in states:
            item = VlmPersonState()
            item.track_id = track_id
            item.person_id = work.visible_track_ids[track_id]
            item.state = state
            output.states.append(item)
        self.states_pub.publish(output)

    def destroy_node(self):
        """Stop the optional worker before ROS destroys its publisher/logger."""
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    """Run the standalone VLM state worker until ROS shutdown."""
    rclpy.init(args=args)
    node = SocialVlmInteraction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
