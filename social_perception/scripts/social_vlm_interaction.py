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


# #!/usr/bin/env python3
# """Block D: turn Block-B person state into a social constraint field.

# The node deliberately has no Nav2 dependency.  It consumes the JSON exported
# by Block B and publishes red Gaussian contours (optional debug grids). This
# keeps the social geometry testable before it is connected to a costmap layer
# or a safety shield.

# ``make_talking_zone`` creates one symmetric Gaussian for exactly two people
# when the experiment is labelled talking. Auto chooses crossing for one track
# and talking for two, solely as a convenience for those two test scenarios;
# it is not a semantic conversation detector. Manual modes remain available.
# All positions are current Block-B measurements in expected_frame.
# Future trajectories are ignored. Activity weights scale cost, not sigma.
# The contours show fractions of each zone's weighted peak, from 90% down to
# 15% by default. Their outer edge is not a truncation of the Gaussian.
# """

# import json
# import math
# from dataclasses import dataclass
# from typing import Iterable, List, Optional, Tuple

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from geometry_msgs.msg import Point
# from nav_msgs.msg import OccupancyGrid
# from visualization_msgs.msg import Marker, MarkerArray


# @dataclass(frozen=True)
# class ConstraintZone:
#     """One current social zone in the tracking frame (odom sim, map real).

#     Front/rear/left/right dimensions are Gaussian standard deviations (m).
#     Contour semi-axes are sigma*sqrt(-2*ln(level)). For a pair,
#     yaw is along the line joining members and front/rear sigmas are equal.
#     """

#     track_id: int
#     x_m: float
#     y_m: float
#     yaw_rad: float
#     front_semi_axis_m: float
#     rear_semi_axis_m: float
#     left_semi_axis_m: float
#     right_semi_axis_m: float
#     # Semantic priority scales peak cost only. It has no units and never
#     # changes the geometric sigmas or the relative-cost contours.
#     social_state: str
#     social_weight: float
#     # Sorted ByteTrack IDs identify a pair independently of input list order.
#     # Empty for individual zones; separation is measured centre-to-centre, m.
#     member_ids: Tuple[int, ...] = ()
#     separation_m: float = 0.0


# class SocialConstraintGrounding(Node):
#     """Block-B String JSON -> current individual/pair MarkerArray in its frame.

#     No TF is invented here: incoming positions must already share
#     expected_frame. Geometry is deterministic and does not invoke a VLM.
#     """

#     def __init__(self) -> None:
#         super().__init__('social_constraint_grounding')

#         # I/O contract. Block B publishes positions in expected_frame (odom sim, map real).
#         self.declare_parameter('input_topic', '/people/tracked_state_json')
#         # Both OccupancyGrid topics carry the same current field. Keep the
#         # legacy /grid name so existing RViz displays show current zones too.
#         self.declare_parameter('current_grid_topic', '/social_constraints/current_grid')
#         self.declare_parameter('combined_grid_topic', '/social_constraints/grid')
#         self.declare_parameter('output_markers_topic', '/social_constraints/markers')
#         self.declare_parameter('expected_frame', 'map')
#         # Default is outlines only, as requested. Enable this explicitly to
#         # compute/publish OccupancyGrid heatmaps for a future cost consumer.
#         self.declare_parameter('publish_cost_grid', False)

#         # Fixed grid in the tracking frame. Keep it fixed so RViz and a future
#         # costmap consumer do not see a jumping origin.
#         self.declare_parameter('grid_resolution_m', 0.05)
#         self.declare_parameter('grid_origin_x_m', -10.0)
#         self.declare_parameter('grid_origin_y_m', -10.0)
#         self.declare_parameter('grid_width_m', 20.0)
#         self.declare_parameter('grid_height_m', 20.0)
#         self.declare_parameter('maximum_cost', 100)
#         self.declare_parameter('minimum_published_cost', 1)
#         self.declare_parameter('combine_mode', 'max')  # max | sum_clamped

#         # Crossing: sigma_h=[1+a*(1-c)]*(d0+k*|v|). Talking pair:
#         # sigma_h=sigma_r=[1+a*(1-c)]*(sep+d0/4), sigma_s=sigma_h/3.
#         # sep is Euclidean centre-to-centre distance in metres. d0 is metres,
#         # k is seconds, a/c dimensionless. c is an explicit experiment value,
#         # not detector confidence; side and rear sigma are sigma_h/3.
#         self.declare_parameter('gaussian_d0_m', 0.5)
#         self.declare_parameter('gaussian_a', 0.5)
#         self.declare_parameter('gaussian_c', 0.9)
#         self.declare_parameter('gaussian_k_s', 0.5)
#         # Dimensionless fractions of EACH zone's weighted peak, not absolute
#         # OccupancyGrid cost. Example: talking peak=90 -> 15% means cost=13.5.
#         # Visual contours only: they do not truncate zone_cost or change sigma.
#         self.declare_parameter('contour_levels', [0.90, 0.75, 0.60, 0.45, 0.30, 0.15])
#         # Auto is scoped to the current tests: one track -> crossing, exactly
#         # two -> talking pair. It does not recognise activities from images,
#         # velocities or Gazebo state. Explicit labels override count selection.
#         self.declare_parameter('social_state', 'auto')
#         # Activity weights are fractions of maximum_cost, not probabilities.
#         # Keep each in [0, 1] so OccupancyGrid stays within its 0..100 contract.
#         self.declare_parameter('social_weight_talking', 0.9)
#         self.declare_parameter('social_weight_waiting', 0.6)
#         self.declare_parameter('social_weight_crossing', 0.5)
#         self.declare_parameter('marker_lifetime_s', 0.75)

#         self.gaussian_d0 = float(self.get_parameter('gaussian_d0_m').value)
#         self.gaussian_a = float(self.get_parameter('gaussian_a').value)
#         self.gaussian_c = float(self.get_parameter('gaussian_c').value)
#         self.gaussian_k = float(self.get_parameter('gaussian_k_s').value)
#         if (not all(math.isfinite(v) for v in (
#                 self.gaussian_d0, self.gaussian_a, self.gaussian_c, self.gaussian_k))
#                 or self.gaussian_d0 <= 0 or self.gaussian_a < 0
#                 or self.gaussian_k < 0 or not 0 <= self.gaussian_c <= 1):
#             raise ValueError('Gaussian requires d0>0, a/k>=0 and 0<=c<=1 (finite)')

#         levels = [float(level) for level in self.get_parameter('contour_levels').value]
#         if not levels or any(not math.isfinite(level) or not 0.0 < level < 1.0
#                              for level in levels):
#             raise ValueError('contour_levels must contain finite fractions between 0 and 1')
#         # Inner -> outer order with no duplicated lines. The smallest level
#         # defines the displayed extent; 15% corresponds to 1.948 sigma.
#         self.contour_levels = sorted(set(levels), reverse=True)
#         self.outer_contour_scale = math.sqrt(-2.0 * math.log(self.contour_levels[-1]))

#         self.social_weights = {
#             state: float(self.get_parameter(f'social_weight_{state}').value)
#             for state in ('talking', 'waiting', 'crossing')
#         }
#         if any(not math.isfinite(weight) or not 0.0 <= weight <= 1.0
#                for weight in self.social_weights.values()):
#             raise ValueError('Social weights must be finite fractions in [0, 1]')
#         self.social_state = str(self.get_parameter('social_state').value).strip().lower()
#         if self.social_state not in ('auto', *self.social_weights):
#             raise ValueError('social_state must be auto, talking, waiting or crossing')

#         self.input_topic = self.get_parameter('input_topic').value
#         self.expected_frame = self.get_parameter('expected_frame').value
#         self.publish_cost_grid = bool(self.get_parameter('publish_cost_grid').value)
#         self.grid_resolution_m = max(0.01, float(self.get_parameter('grid_resolution_m').value))
#         self.grid_origin_x_m = float(self.get_parameter('grid_origin_x_m').value)
#         self.grid_origin_y_m = float(self.get_parameter('grid_origin_y_m').value)
#         self.grid_width_m = max(self.grid_resolution_m, float(self.get_parameter('grid_width_m').value))
#         self.grid_height_m = max(self.grid_resolution_m, float(self.get_parameter('grid_height_m').value))
#         self.maximum_cost = max(1, min(100, int(self.get_parameter('maximum_cost').value)))
#         self.minimum_published_cost = max(1, min(self.maximum_cost,
#                                                   int(self.get_parameter('minimum_published_cost').value)))
#         self.combine_mode = self.get_parameter('combine_mode').value

#         self.grid_width_cells = int(math.ceil(self.grid_width_m / self.grid_resolution_m))
#         self.grid_height_cells = int(math.ceil(self.grid_height_m / self.grid_resolution_m))

#         # Optional output: nav_msgs/OccupancyGrid (integer cost 0..100), in
#         # expected_frame. No grid publishers or raster work in outline-only
#         # mode; disable old RViz Map displays to remove their cached heatmaps.
#         if self.publish_cost_grid:
#             self.current_grid_pub = self.create_publisher(
#                 OccupancyGrid, self.get_parameter('current_grid_topic').value, 1)
#             self.combined_grid_pub = self.create_publisher(
#                 OccupancyGrid, self.get_parameter('combined_grid_topic').value, 1)
#         # RViz output: visualization_msgs/MarkerArray, positions in metres,
#         # in expected_frame. Input std_msgs/String carries current Block-B JSON.
#         self.markers_pub = self.create_publisher(
#             MarkerArray, self.get_parameter('output_markers_topic').value, 1)
#         self.subscription = self.create_subscription(
#             String, self.input_topic, self.block_b_callback, 10)

#         self.get_logger().info(
#             f'Block D ready: {self.input_topic} -> '
#             f'{self.get_parameter("output_markers_topic").value} (current only), '
#             f'expected_frame={self.expected_frame}, '
#             f'state_selection={self.social_state}, '
#             f'publish_cost_grid={self.publish_cost_grid}, contours={self.contour_levels}')

#     def block_b_callback(self, message: String) -> None:
#         """Use one String snapshot to update outlines and optional grids.

#         Header time/frame are preserved. DELETEALL clears the old pair when
#         counts change. Auto may then show the remaining track as crossing;
#         manual talking waits until both people are present again.
#         """
#         try:
#             state = json.loads(message.data)
#             frame_id = self.validate_block_b(state)
#             zones = self.build_constraint_zones(state.get('people', []))
#         except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
#             self.get_logger().warn(f'Ignoring invalid Block-B state: {error}')
#             return

#         stamp_ns = int(state.get('header', {}).get('stamp_ns', 0))
#         # Optional grids share the current field; default only draws outlines.
#         if self.publish_cost_grid:
#             grid = self.make_grid(frame_id, stamp_ns, zones)
#             self.current_grid_pub.publish(grid)
#             self.combined_grid_pub.publish(grid)
#         self.markers_pub.publish(self.make_markers(frame_id, stamp_ns, zones))

#     def validate_block_b(self, state: dict) -> str:
#         """Validate the small stable contract D needs from B."""
#         header = state.get('header')
#         if not isinstance(header, dict):
#             raise ValueError('missing header')
#         frame_id = header.get('frame_id')
#         if frame_id != self.expected_frame:
#             raise ValueError(
#                 f'expected frame {self.expected_frame!r}, received {frame_id!r}; '
#                 'transform in Block B before sending to D')
#         people = state.get('people')
#         if not isinstance(people, list):
#             raise ValueError('people must be a list')
#         return frame_id

#     def build_constraint_zones(self, people: Iterable[dict]) -> List[ConstraintZone]:
#         """Choose one talking pair OR individual zones for this experiment.

#         Auto dispatches only the one-person and two-person test cases, using
#         the current input count. It intentionally does not choose pairs from
#         a crowd (>2) or infer talking from proximity. Brief occlusion can
#         change a pair to individual geometry; use manual talking to require
#         both members throughout. No history or extra future zones are kept.
#         """
#         people = list(people)
#         selected_state = self.social_state
#         if selected_state == 'auto':
#             if len(people) not in (1, 2):
#                 return []
#             selected_state = 'talking' if len(people) == 2 else 'crossing'
#         if selected_state == 'talking':
#             if len(people) != 2:
#                 return []
#             try:
#                 return [self.make_talking_zone(people[0], people[1])]
#             except (TypeError, ValueError, KeyError) as error:
#                 self.get_logger().warn(f'Skipping invalid talking pair: {error}')
#                 return []
#         zones: List[ConstraintZone] = []
#         for person in people:
#             try:
#                 zones.extend(self.make_zones_for_person(person))
#             except (TypeError, ValueError, KeyError) as error:
#                 self.get_logger().warn(f'Skipping malformed person: {error}')
#         return zones

#     def make_talking_zone(self, first: dict, second: dict) -> ConstraintZone:
#         """Two current Block-B positions -> one symmetric talking Gaussian.

#         position_m is in expected_frame (metres); track_id is a ByteTrack ID.
#         Group centre is the midpoint. atan2(dy, dx) orients the long axis
#         along the pair, independent of unreliable body yaw for still people.
#         sep=hypot(dx,dy) is the FULL centre-to-centre separation, not sep/2.
#         The user's formula is factor*(sep+d0/4), not factor*(sep+d0)/4.
#         Front/rear symmetry means swapping people cannot change the field.
#         """
#         # Sorting fixes the displayed identity/yaw when B changes list order.
#         first, second = sorted((first, second), key=lambda p: int(p['track_id']))
#         first_id, second_id = int(first['track_id']), int(second['track_id'])
#         x1, y1 = float(first['position_m']['x']), float(first['position_m']['y'])
#         x2, y2 = float(second['position_m']['x']), float(second['position_m']['y'])
#         if first_id == second_id or not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
#             raise ValueError('pair needs distinct track IDs and finite positions')
#         dx, dy = x2 - x1, y2 - y1
#         separation_m = math.hypot(dx, dy)
#         # Coincident centres have no pair axis; do not draw an invented yaw.
#         if not math.isfinite(separation_m) or separation_m <= 1e-6:
#             raise ValueError('pair centres must be separated by more than 1e-6 m')
#         factor = 1.0 + self.gaussian_a * (1.0 - self.gaussian_c)
#         sigma_h = sigma_r = factor * (separation_m + self.gaussian_d0 / 4.0)
#         sigma_s = sigma_h / 3.0
#         return ConstraintZone(
#             first_id, x1 / 2.0 + x2 / 2.0, y1 / 2.0 + y2 / 2.0,
#             math.atan2(dy, dx), sigma_h, sigma_r, sigma_s, sigma_s,
#             'talking', self.social_weights['talking'],
#             (first_id, second_id), separation_m)

#     # ------------------------------------------------------------------
#     # Crossing geometry from Block-B velocity and configured coefficients.
#     # Inputs are current tracking-frame position/velocity/body yaw.
#     # Output stays a list of ConstraintZone, so grid/RViz need no rewiring.
#     # ------------------------------------------------------------------
#     def make_zones_for_person(self, person: dict) -> List[ConstraintZone]:
#         """Use tracking-frame velocity (m/s) for individual size and heading.

#         Auto reaches this method only for the one-person crossing case;
#         resolve its actual weight without changing the configured selector.
#         """
#         track_id = int(person['track_id'])
#         position = person['position_m']
#         velocity = person.get('velocity_mps', {})
#         x_m = float(position['x'])
#         y_m = float(position['y'])
#         vx_mps = float(velocity.get('vx', 0.0))
#         vy_mps = float(velocity.get('vy', 0.0))
#         speed_mps = math.hypot(vx_mps, vy_mps)
#         if not all(math.isfinite(v) for v in (x_m, y_m, vx_mps, vy_mps, speed_mps)):
#             raise ValueError('non-finite position or velocity')
#         # Crossing front follows velocity, including sideways walking. At rest
#         # use valid body yaw; without any heading, skip rather than invent one.
#         if speed_mps > 1e-6:
#             yaw_rad = math.atan2(vy_mps, vx_mps)
#         elif bool(person.get('body_orientation_valid', False)):
#             yaw_rad = float(person['body_orientation_rad'])
#         elif bool(person.get('motion_heading_valid', False)):
#             yaw_rad = float(person['motion_heading_rad'])
#         else:
#             return []
#         if not math.isfinite(yaw_rad):
#             raise ValueError('non-finite heading')

#         # Deliberately ignore prediction.trajectory_m: D visualises only the
#         # measured current position, while speed still controls Gaussian size.
#         selected_state = 'crossing' if self.social_state == 'auto' else self.social_state
#         return [self.make_crossing_zone(
#             track_id, x_m, y_m, yaw_rad, speed_mps, selected_state)]

#     def make_crossing_zone(
#             self, track_id: int, x_m: float, y_m: float,
#             yaw_rad: float, speed_mps: float, social_state: str) -> ConstraintZone:
#         """Compute sigma (m) and activity priority using the same geometry."""
#         sigma_h = ((1.0 + self.gaussian_a * (1.0 - self.gaussian_c)) *
#                    (self.gaussian_d0 + self.gaussian_k * speed_mps))
#         sigma_s = sigma_r = sigma_h / 3.0
#         return ConstraintZone(track_id, x_m, y_m, yaw_rad,
#                               sigma_h, sigma_r, sigma_s, sigma_s,
#                               social_state, self.social_weights[social_state])

#     # ------------------------------------------------------------------
#     # POLICY HOOK 2: replace this evaluator for a different K_soc(x, y, t).
#     # It receives a cell and one current social zone and returns [0, 100].
#     # ------------------------------------------------------------------
#     def zone_cost(self, x_m: float, y_m: float, zone: ConstraintZone) -> int:
#         """Evaluate maximum_cost*w_state*exp(-q/2) in the person's axes.

#         q=(forward/sigma_front_or_rear)^2+(left/sigma_side)^2.
#         There is no cutoff: displayed contours are visual references only.
#         Only optional grid rasterisation rounds small tail costs to zero.
#         """
#         dx = x_m - zone.x_m
#         dy = y_m - zone.y_m
#         forward = math.cos(zone.yaw_rad) * dx + math.sin(zone.yaw_rad) * dy
#         lateral = -math.sin(zone.yaw_rad) * dx + math.cos(zone.yaw_rad) * dy
#         forward_axis = zone.front_semi_axis_m if forward >= 0.0 else zone.rear_semi_axis_m
#         lateral_axis = zone.left_semi_axis_m if lateral >= 0.0 else zone.right_semi_axis_m
#         normalized_distance_sq = (
#             (forward / forward_axis) ** 2 +
#             (lateral / lateral_axis) ** 2)

#         return int(round(self.maximum_cost * zone.social_weight *
#                          math.exp(-0.5 * normalized_distance_sq)))

#     def make_grid(self, frame_id: str, stamp_ns: int,
#                   zones: List[ConstraintZone]) -> OccupancyGrid:
#         """Optional current cost raster sampled at cell centres in metres.

#         maximum_cost*w*exp(-q/2) is rounded to 0..100; max or sum_clamped
#         combines zones. The finite grid extent is not a Gaussian boundary.
#         """
#         grid = OccupancyGrid()
#         grid.header.frame_id = frame_id
#         grid.header.stamp.sec = stamp_ns // 1_000_000_000
#         grid.header.stamp.nanosec = stamp_ns % 1_000_000_000
#         grid.info.resolution = self.grid_resolution_m
#         grid.info.width = self.grid_width_cells
#         grid.info.height = self.grid_height_cells
#         grid.info.origin.position.x = self.grid_origin_x_m
#         grid.info.origin.position.y = self.grid_origin_y_m
#         grid.info.origin.orientation.w = 1.0
#         grid.data = [0] * (self.grid_width_cells * self.grid_height_cells)

#         for row in range(self.grid_height_cells):
#             y_m = self.grid_origin_y_m + (row + 0.5) * self.grid_resolution_m
#             for col in range(self.grid_width_cells):
#                 x_m = self.grid_origin_x_m + (col + 0.5) * self.grid_resolution_m
#                 costs = [self.zone_cost(x_m, y_m, zone) for zone in zones]
#                 if not costs:
#                     continue
#                 if self.combine_mode == 'sum_clamped':
#                     cost = min(self.maximum_cost, sum(costs))
#                 else:
#                     cost = max(costs)
#                 if cost >= self.minimum_published_cost:
#                     grid.data[row * self.grid_width_cells + col] = cost
#         return grid

#     def make_markers(self, frame_id: str, stamp_ns: int,
#                      zones: List[ConstraintZone]) -> MarkerArray:
#         """Draw red iso-cost LINE_STRIPs for individuals and talking pairs.

#         For level alpha: exp(-q/2)=alpha -> q=-2*ln(alpha), hence each
#         semi-axis is sigma*sqrt(-2*ln(alpha)), in metres in frame_id.
#         TEXT_VIEW_FACING labels state the percentage of this zone's peak;
#         the main label separates sigma from outer contour dimensions.
#         DELETEALL and finite lifetime clear old pairs/levels when input changes.
#         """
#         markers = MarkerArray()
#         clear = Marker()
#         clear.action = Marker.DELETEALL
#         markers.markers.append(clear)
#         lifetime_s = float(self.get_parameter('marker_lifetime_s').value)

#         for zone_id, zone in enumerate(zones):
#             # Zero weight has no field to contour; DELETEALL removes old lines.
#             if zone.social_weight == 0.0:
#                 continue
#             for level_id, level in enumerate(self.contour_levels):
#                 contour_scale = math.sqrt(-2.0 * math.log(level))
#                 marker = Marker()
#                 marker.header.frame_id = frame_id
#                 marker.header.stamp.sec = stamp_ns // 1_000_000_000
#                 marker.header.stamp.nanosec = stamp_ns % 1_000_000_000
#                 marker.ns = ('social_constraint_talking_group' if zone.member_ids
#                              else 'social_constraint_current')
#                 # One namespace can contain several people and several levels;
#                 # a unique ID prevents RViz replacing one line with another.
#                 marker.id = zone_id * len(self.contour_levels) + level_id
#                 marker.type = Marker.LINE_STRIP
#                 marker.action = Marker.ADD
#                 marker.pose.orientation.w = 1.0
#                 marker.scale.x = 0.03 if level_id == len(self.contour_levels)-1 else 0.015
#                 marker.color.a = 0.85
#                 marker.color.r = 1.0
#                 marker.color.g = 0.05
#                 marker.color.b = 0.0
#                 marker.lifetime.sec = int(lifetime_s)
#                 marker.lifetime.nanosec = int((lifetime_s % 1.0) * 1_000_000_000)

#                 # Local front/side axes -> tracking coordinates; 72 segments
#                 # keep large pair contours smooth. No time prediction is used.
#                 for index in range(73):
#                     angle = 2.0 * math.pi * index / 72.0
#                     forward_axis = (zone.front_semi_axis_m if math.cos(angle) >= 0.0
#                                     else zone.rear_semi_axis_m)
#                     lateral_axis = (zone.left_semi_axis_m if math.sin(angle) >= 0.0
#                                     else zone.right_semi_axis_m)
#                     forward = contour_scale * forward_axis * math.cos(angle)
#                     lateral = contour_scale * lateral_axis * math.sin(angle)
#                     point = Point()
#                     point.x = zone.x_m + math.cos(zone.yaw_rad)*forward - math.sin(zone.yaw_rad)*lateral
#                     point.y = zone.y_m + math.sin(zone.yaw_rad)*forward + math.cos(zone.yaw_rad)*lateral
#                     point.z = 0.05
#                     marker.points.append(point)
#                 markers.markers.append(marker)

#                 # Spread labels around the contours rather than stacking six
#                 # values at the same heading; these are peak fractions, not m.
#                 level_label = Marker()
#                 level_label.header = marker.header
#                 level_label.ns = 'social_constraint_contour_labels'
#                 level_label.id = marker.id
#                 level_label.type = Marker.TEXT_VIEW_FACING
#                 level_label.action = Marker.ADD
#                 level_label.pose.position = marker.points[(4 + 9*level_id) % 72]
#                 level_label.pose.orientation.w = 1.0
#                 level_label.scale.z = 0.10
#                 level_label.color.r = level_label.color.g = level_label.color.b = 1.0
#                 level_label.color.a = 1.0
#                 level_label.lifetime = marker.lifetime
#                 level_label.text = f'{100*level:g}%'
#                 markers.markers.append(level_label)

#             # Group labels show both member IDs and measured separation. Rear
#             # and side sigma differ in talking geometry, so list them separately.
#             label = Marker()
#             label.header = marker.header
#             label.ns = 'social_constraint_sigma'
#             label.id = zone_id
#             label.type = Marker.TEXT_VIEW_FACING
#             label.action = Marker.ADD
#             label.pose.position.x = zone.x_m
#             label.pose.position.y = zone.y_m
#             label.pose.position.z = 1.8
#             label.pose.orientation.w = 1.0
#             label.scale.z = 0.18
#             label.color.r = label.color.g = label.color.b = label.color.a = 1.0
#             label.lifetime = marker.lifetime
#             identity = (f'PAIR {zone.member_ids[0]} + {zone.member_ids[1]} '
#                         f'| sep={zone.separation_m:.2f} m' if zone.member_ids
#                         else f'ID {zone.track_id}')
#             length = self.outer_contour_scale * (zone.front_semi_axis_m + zone.rear_semi_axis_m)
#             width = self.outer_contour_scale * (zone.left_semi_axis_m + zone.right_semi_axis_m)
#             label.text = (
#                 f'{identity} | outer {100*self.contour_levels[-1]:g}% peak\n'
#                 f'{zone.social_state} w={zone.social_weight:g} '
#                 f'peak={self.maximum_cost*zone.social_weight:g}\n'
#                 f'sigma: h={zone.front_semi_axis_m:.2f} m '
#                 f'r={zone.rear_semi_axis_m:.2f} m s={zone.left_semi_axis_m:.2f} m\n'
#                 f'outer: length={length:.2f} m width={width:.2f} m')
#             markers.markers.append(label)
#         return markers


# def main(args: Optional[List[str]] = None) -> None:
#     rclpy.init(args=args)
#     node = SocialConstraintGrounding()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()