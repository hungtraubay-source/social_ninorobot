#!/usr/bin/env python3
"""Localize people from RGB-D with YOLO detection."""

import copy
import contextlib
import json
import math
import queue
import re
import threading
import time
import unicodedata
import warnings
from collections import deque
from enum import IntEnum
from pathlib import Path

# Keep third-party cosmetic warnings from hiding the two operator-facing
# messages that matter: model-load progress and the conversation decision.
warnings.filterwarnings('ignore', message='Unable to import Axes3D.*')
warnings.filterwarnings('ignore', message='`max_length` is ignored.*')
warnings.filterwarnings('ignore', message='You passed `quantization_config`.*')

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Empty, String
from tf2_ros import Buffer, TransformException, TransformListener
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from social_perception.msg import (
    Keypoint2D,
    Keypoint3D,
    People,
    PeopleObservations,
    PeopleOrientations,
    Person,
    PersonObservation,
    PersonOrientation,
    TalkingInteraction,
    TalkingInteractions,
)


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def rotate_vector(vector, quaternion):
    """Rotate a 3-D vector by a geometry_msgs Quaternion."""
    x, y, z = vector
    qx, qy, qz, qw = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + qy * tz - qz * ty,
        y + qw * ty + qz * tx - qx * tz,
        z + qw * tz + qx * ty - qy * tx,
    )


def yaw_quaternion(yaw):
    """Quaternion for a yaw-only orientation in the navigation frame."""
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def wrap_angle(angle):
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def normalized_text(value):
    text = unicodedata.normalize('NFD', str(value).lower())
    text = ''.join(character for character in text
                   if unicodedata.category(character) != 'Mn')
    return text.replace('đ', 'd')


def talking_from_response(response):
    """Parse the model's Vietnamese/English JSON, conservatively."""
    keyed_values = []
    match = re.search(r'\{.*\}', response, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))

            def collect(item, key=''):
                if isinstance(item, dict):
                    for child_key, child in item.items():
                        collect(child, str(child_key))
                elif isinstance(item, list):
                    for child in item:
                        collect(child, key)
                elif any(token in normalized_text(key) for token in (
                        'talk', 'noi chuyen', 'tro chuyen', 'interaction', 'state')):
                    keyed_values.append((normalized_text(key), item))

            collect(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    # A dedicated talking field is authoritative. Generic state/interaction
    # fields in the training JSON may describe an individual, not the pair.
    talking_values = [value for key, value in keyed_values
                      if any(token in key for token in (
                          'talk', 'noi chuyen', 'tro chuyen'))]
    values = talking_values or [value for _, value in keyed_values] or [response]

    for value in values:
        if value is False or (isinstance(value, (int, float)) and value == 0):
            return False, 0.95
        text = normalized_text(value)
        if any(token in text for token in (
                'khong', 'false', 'not talking', 'no conversation', 'khong xac dinh')):
            return False, 0.90
        # A bare "no", mirroring the bare "co" accepted below. It cannot be a
        # substring test: "no" also sits inside "noi chuyen", which is a yes.
        if text.strip() in ('no', 'not'):
            return False, 0.90
    for value in values:
        if value is True or (isinstance(value, (int, float)) and value == 1):
            return True, 0.95
        text = normalized_text(value)
        if any(token in text for token in (
                'dang noi chuyen', 'dang tro chuyen', 'co noi chuyen',
                'talking', 'conversation', 'true', 'yes')) or text.strip() == 'co':
            return True, 0.90
    return False, 0.0


# ---------------------------------------------------------------------------
# LEGACY / DISABLED VLM SUPPORT
#
# The classes and helpers in this section are retained only so an old trained
# Qwen adapter can be restored deliberately in a future experiment.  The
# runtime path below forces ``self.vlm_enabled`` to False, does not start this
# worker, and does not enqueue RGB crops.  Nav2 must not depend on this code.
# ---------------------------------------------------------------------------


class VlmProgress:
    """Log stage-based VLM progress while blocking native operations run.

    Transformers does not expose byte-level progress callbacks for cached
    checkpoints. Percentages therefore describe completed initialization
    stages. A heartbeat repeats the current stage and elapsed time so a long
    ``from_pretrained`` call is visibly alive without pretending to know its
    remaining duration.
    """

    def __init__(self, logger, label, initial_stage,
                 heartbeat_seconds=10.0, bar_width=20):
        self.logger = logger
        self.label = label
        self.heartbeat_seconds = heartbeat_seconds
        self.bar_width = bar_width
        self.started_at = time.monotonic()
        self.percent = 0
        self.stage = initial_stage
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self._log()
        self.thread = threading.Thread(
            target=self._heartbeat,
            name='vlm-load-progress',
            daemon=True,
        )
        self.thread.start()

    def update(self, percent, stage):
        with self.lock:
            self.percent = max(self.percent, min(100, int(percent)))
            self.stage = stage
        self._log()

    def complete(self, stage):
        self.update(100, stage)
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def fail(self, error):
        with self.lock:
            self.stage = f'FAILED: {type(error).__name__}: {error}'
        self._log()
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _heartbeat(self):
        while not self.stop_event.wait(self.heartbeat_seconds):
            self._log()

    def _log(self):
        with self.lock:
            percent = self.percent
            stage = self.stage
        completed = int(round(self.bar_width * percent / 100.0))
        bar = '#' * completed + '-' * (self.bar_width - completed)
        elapsed = time.monotonic() - self.started_at
        try:
            self.logger.info(
                f'{self.label} [{bar}] {percent:3d}% | {stage} | '
                f'elapsed={elapsed:.0f}s')
        except Exception:
            # Shutdown may invalidate the ROS context while a native model
            # loader is still returning from a background thread.
            pass


class VlmBackend:
    """Standard Transformers/PEFT loader for the supplied Qwen2-VL adapter."""

    # Generation, not the image, dominates latency on a small GPU: measured on
    # a Quadro T1000 the prefill costs ~1.8 s and every further decode step
    # ~0.36 s, so the 7-token answer `{"talking":"Không"}` spent ~2.2 s writing
    # punctuation the prompt already dictates. Teacher-forcing that punctuation
    # leaves only the decisive word to generate.
    ANSWER_PREFIX = '{"talking":"'
    # Enough for the longest tokenization of either word ('KH','Ô','NG'); the
    # shorter ones simply run into the closing quote, which is cut off below.
    PREFIX_ANSWER_TOKENS = 3

    def __init__(self, adapter_path, base_model, load_in_4bit, require_cuda,
                 max_new_tokens, min_pixels, max_pixels, logger,
                 execution_lock=None, force_answer_prefix=True):
        self.logger = logger
        self.execution_lock = execution_lock
        self.force_answer_prefix = force_answer_prefix
        progress = VlmProgress(
            logger, 'VLM LOAD', 'Starting VLM initialization')
        progress.start()
        try:
            progress.update(5, 'Checking Pillow compatibility')
            # Ubuntu 22.04 provides Pillow 9.0.1, whose resampling constants
            # live directly on PIL.Image. New Transformers expects the enum
            # introduced in Pillow 9.1. This alias is API-compatible and
            # avoids a misleading downstream PEFT import error.
            from PIL import Image as PilImageModule
            if not hasattr(PilImageModule, 'Resampling'):
                class PillowResampling(IntEnum):
                    NEAREST = PilImageModule.NEAREST
                    LANCZOS = PilImageModule.LANCZOS
                    BILINEAR = PilImageModule.BILINEAR
                    BICUBIC = PilImageModule.BICUBIC
                    BOX = PilImageModule.BOX
                    HAMMING = PilImageModule.HAMMING

                PilImageModule.Resampling = PillowResampling

            progress.update(10, 'Importing PyTorch')
            import torch
            progress.update(20, 'Importing PEFT and Transformers')
            from peft import PeftModel
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from transformers.utils import logging as transformers_logging
            transformers_logging.set_verbosity_error()
            progress.update(30, 'ML libraries imported')

            self.torch = torch
            self.max_new_tokens = max_new_tokens
            self.min_pixels = min_pixels
            self.max_pixels = max_pixels
            use_cuda = torch.cuda.is_available()
            progress.update(
                32, f'Runtime ready: CUDA={use_cuda}, CUDA runtime={torch.version.cuda}')
            if require_cuda and not use_cuda:
                raise RuntimeError(
                    'CUDA is required for VLM inference but is unavailable')
            model_kwargs = {
                'device_map': 'auto' if use_cuda else None,
                'torch_dtype': torch.float16 if use_cuda else torch.float32,
            }
            if load_in_4bit and use_cuda:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif load_in_4bit:
                logger.warn(
                    'CUDA is unavailable; VLM will attempt a local CPU fallback, '
                    'which can be very slow for Qwen2-VL')

            progress.update(35, f'Loading Qwen2-VL base model: {base_model}')
            base = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model, **model_kwargs)
            device_map = getattr(base, 'hf_device_map', {})
            offloaded_devices = sorted({
                str(device) for device in device_map.values()
                if str(device) in ('cpu', 'disk')
            })
            if offloaded_devices:
                logger.warn(
                    'Part of the VLM was offloaded to '
                    f'{", ".join(offloaded_devices)}; inference will be slower')
            progress.update(75, 'Base model loaded; attaching LoRA adapter')
            self.model = PeftModel.from_pretrained(base, str(adapter_path))
            self.model.eval()
            progress.update(88, 'LoRA adapter attached; loading processor')
            processor_kwargs = {
                'min_pixels': self.min_pixels,
                'max_pixels': self.max_pixels,
            }
            try:
                self.processor = AutoProcessor.from_pretrained(
                    str(adapter_path), **processor_kwargs)
            except (OSError, ValueError):
                progress.update(92, 'Loading processor from base model')
                self.processor = AutoProcessor.from_pretrained(
                    base_model, **processor_kwargs)
            progress.update(97, 'Processor loaded; selecting inference device')
            self.input_device = next(self.model.parameters()).device
            progress.update(98, 'Warming up CUDA kernels')
            self._warm_up()
            progress.complete(
                f'VLM READY: adapter loaded on {self.input_device}')
        except Exception as error:
            progress.fail(error)
            raise

    def _warm_up(self):
        """Pay the one-off CUDA/bitsandbytes kernel cost before the first pair.

        The first generate() call measured 9.4 s against 4.1 s for every call
        after it. Spending that here means the first real conversation is
        judged at the steady-state latency instead of waiting out kernel
        autotuning while two people stand in front of the camera.
        """
        started_at = time.monotonic()
        try:
            blank = np.full((256, 384, 3), 127, dtype=np.uint8)
            self.infer(blank, 'warm-up')
        except Exception as error:  # A failed warm-up must not block startup.
            self.logger.warn(
                f'VLM warm-up failed ({type(error).__name__}: {error}); '
                'the first inference will be slower')
            return
        self.logger.info(
            f'VLM warm-up finished in {time.monotonic() - started_at:.2f}s')

    def infer(self, bgr_image, prompt, pair=None):
        from PIL import Image as PilImage

        started_at = time.monotonic()
        image = PilImage.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        max_new_tokens = self.max_new_tokens
        if self.force_answer_prefix:
            # Start the assistant turn mid-answer so the model resumes from the
            # forced prefix. It only has to produce the word, which also ends
            # the malformed replies (`{"talking":"có}`, `{"talking":true}`)
            # that used to cost a whole inference to recover from.
            text += self.ANSWER_PREFIX
            max_new_tokens = self.PREFIX_ANSWER_TOKENS
        # This is a one-item batch, so padding is unnecessary and only causes
        # a noisy Transformers max_length warning.
        inputs = self.processor(
            text=[text], images=[image], padding=False, return_tensors='pt')
        inputs = {key: value.to(self.input_device) for key, value in inputs.items()}
        lock_context = (self.execution_lock if self.execution_lock is not None
                        else contextlib.nullcontext())
        # Split the wait into preprocessing, waiting for the lock and the model
        # itself. The same crop takes seconds standalone and far longer in the
        # running node, and only a breakdown says which of the three grew.
        prepared_at = time.monotonic()
        with lock_context:
            acquired_at = time.monotonic()
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        self.last_timing = (prepared_at - started_at,
                            acquired_at - prepared_at,
                            time.monotonic() - acquired_at,
                            int(inputs['input_ids'].shape[1]))
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()
        if self.force_answer_prefix:
            # Reattach what was teacher-forced so the published response stays
            # the same JSON object every consumer already parses and logs. The
            # word is the model's; only the syntax around it was supplied.
            word = answer.split('"')[0].strip()
            answer = self.ANSWER_PREFIX + word + '"}'
        return answer


class SocialVlmPerception(Node):
    def __init__(self):
        super().__init__('social_vlm_perception')
        self._declare_parameters()

        self.confidence = float(self.get_parameter('yolo_confidence').value)
        self.image_size = int(self.get_parameter('yolo_image_size').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.max_depth_age = float(self.get_parameter('maximum_depth_age').value)
        self.min_depth = float(self.get_parameter('minimum_depth').value)
        self.max_depth = float(self.get_parameter('maximum_depth').value)
        self.bytetrack_config = self._resolve_file(
            str(self.get_parameter('bytetrack_config').value))
        if self.bytetrack_config is None:
            raise FileNotFoundError(
                'ByteTrack config not found: '
                f"{self.get_parameter('bytetrack_config').value}")
        self.duplicate_distance = float(
            self.get_parameter('duplicate_merge_distance').value)
        self.track_timeout = float(self.get_parameter('tracking_timeout').value)
        self.velocity_alpha = float(self.get_parameter('velocity_smoothing').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.max_odom_age = float(self.get_parameter('maximum_odom_age').value)
        self.prediction_horizon = float(
            self.get_parameter('prediction_horizon_s').value)
        self.prediction_step = float(self.get_parameter('prediction_step_s').value)
        self.prediction_min_speed = float(
            self.get_parameter('prediction_min_speed_mps').value)
        self.prediction_marker_diameter = max(0.01, float(
            self.get_parameter('prediction_marker_diameter_m').value))
        self.keypoint_confidence = float(
            self.get_parameter('keypoint_confidence').value)
        self.keypoint_depth_window = max(
            1, int(self.get_parameter('keypoint_depth_window_px').value))
        self.minimum_shoulder_width = float(
            self.get_parameter('minimum_shoulder_width_m').value)
        self.orientation_keypoint_confidence = float(
            self.get_parameter('orientation_keypoint_confidence').value)
        self.orientation_nose_confidence = float(
            self.get_parameter('orientation_nose_confidence').value)
        self.orientation_face_confidence = float(
            self.get_parameter('orientation_face_confidence').value)
        self.minimum_face_direction = float(
            self.get_parameter('minimum_face_direction_m').value)
        self.minimum_face_alignment = min(1.0, max(0.0, float(
            self.get_parameter('minimum_face_alignment').value)))
        self.orientation_flip_frames = max(
            1, int(self.get_parameter('orientation_flip_frames').value))
        # -1 means keep the last valid heading for the lifetime of its track.
        # A non-negative value keeps the old bounded-by-frames behaviour.
        self.orientation_hold_frames = int(
            self.get_parameter('orientation_hold_frames').value)
        self.maximum_shoulder_width = float(
            self.get_parameter('maximum_shoulder_width_m').value)
        self.minimum_hip_width = float(
            self.get_parameter('minimum_hip_width_m').value)
        self.maximum_hip_width = float(
            self.get_parameter('maximum_hip_width_m').value)
        self.body_yaw_smoothing = min(1.0, max(0.0, float(
            self.get_parameter('body_yaw_smoothing').value)))
        self.orientation_arrow_length = max(0.05, float(
            self.get_parameter('orientation_marker_arrow_length_m').value))
        self.orientation_arrow_lifetime = max(0.1, float(
            self.get_parameter('orientation_marker_lifetime_s').value))
        self.orientation_arrow_color = tuple(
            min(1.0, max(0.0, float(self.get_parameter(
                f'orientation_marker_arrow_color_{channel}').value)))
            for channel in ('r', 'g', 'b'))
        if self.prediction_horizon <= 0.0 or self.prediction_step <= 0.0:
            raise ValueError('prediction_horizon_s and prediction_step_s must be positive.')
        self.yolo_device = str(self.get_parameter('yolo_device').value)
        requested_vlm = bool(self.get_parameter('enable_vlm').value)
        # Conversation semantics now belongs to social_vlm_interaction.py. Keep
        # this deprecated parameter only so old YAML files fail safely instead
        # of making Block B load a GPU model or delaying camera tracking.
        self.vlm_enabled = False
        if requested_vlm:
            self.get_logger().warn(
                'Bỏ enable_vlm ở Block B; chạy node social_vlm_interaction riêng.')
        self.vlm_interval = float(self.get_parameter('vlm_inference_interval').value)
        self.vlm_refresh_interval = float(
            self.get_parameter('vlm_refresh_interval').value)
        self.vlm_position_threshold = float(
            self.get_parameter('vlm_position_change_threshold').value)
        self.max_pair_distance = float(self.get_parameter('maximum_talking_distance').value)
        self.max_vlm_pairs = int(self.get_parameter('maximum_vlm_pairs').value)
        self.negatives_to_clear = max(
            1, int(self.get_parameter('negative_answers_to_clear').value))
        self.interaction_timeout = float(self.get_parameter('interaction_timeout').value)
        self.crop_margin = float(self.get_parameter('vlm_crop_margin').value)
        self.last_vlm_enqueue = 0.0
        # This lock is used only if YOLO is explicitly put on CUDA. The default
        # CPU detector keeps processing current camera frames while Qwen uses
        # the GPU in the background.
        self.ml_execution_lock = threading.Lock()

        yolo_path = self._resolve_file(str(self.get_parameter('yolo_model_path').value))
        if yolo_path is None:
            yolo_path = str(self.get_parameter('yolo_model_path').value)
            self.get_logger().warn(
                f'YOLO model {yolo_path} is not local; Ultralytics may download it')
        self.yolo = YOLO(str(yolo_path))
        self.pose_model_warning_logged = False
        self.depth_msg = None
        self.camera_info = None
        self.depth_camera_info = None
        self.latest_odom = None
        # YOLO may process an RGB frame after newer sensor messages arrived.
        # Keep a short timestamped history and select the matching depth/odom
        # instead of pairing that RGB frame with whichever message arrived last.
        self.sensor_lock = threading.Lock()
        self.depth_buffer = deque(maxlen=120)
        self.odom_buffer = deque(maxlen=240)
        self.rgb_frames = 0
        self.depth_frames = 0
        self.tracks = {}
        self.tracks_generation = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_lock = threading.Lock()
        self.latest_people = {}
        self.latest_header = None
        self.latest_observation_sequence = 0
        self.scene_generation = 0
        self.interaction_cache = {}
        self.vlm_inflight_pairs = {}
        self.last_vlm_scenes = {}
        self.last_logged_decisions = {}
        self.last_interactions_signature = None
        self.last_interactions_publish = 0.0
        self.work_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.vlm_thread = None

        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)
        info_topic = str(self.get_parameter('camera_info_topic').value)
        depth_info_topic = str(
            self.get_parameter('depth_camera_info_topic').value)
        # YOLO inference is intentionally isolated from the lightweight depth
        # callbacks. A single-threaded executor made depth wait behind YOLO,
        # producing artificial RGB/depth timestamp mismatches.
        self.rgb_callback_group = MutuallyExclusiveCallbackGroup()
        self.depth_callback_group = MutuallyExclusiveCallbackGroup()
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, depth_topic, self.depth_callback,
                                 camera_qos,
                                 callback_group=self.depth_callback_group)
        self.create_subscription(CameraInfo, info_topic, self.info_callback,
                                 camera_qos,
                                 callback_group=self.depth_callback_group)
        if depth_info_topic:
            self.create_subscription(
                CameraInfo, depth_info_topic, self.depth_info_callback,
                camera_qos, callback_group=self.depth_callback_group)
        self.create_subscription(Image, rgb_topic, self.rgb_callback,
                                 camera_qos,
                                 callback_group=self.rgb_callback_group)
        odom_topic = str(self.get_parameter('odom_topic').value)
        if odom_topic:
            self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        clear_topic = str(self.get_parameter('clear_interactions_topic').value)
        if clear_topic:
            self.create_subscription(
                Empty, clear_topic, self.clear_interactions_callback, 10)
        track_reset_topic = str(self.get_parameter('track_reset_topic').value)
        if track_reset_topic and track_reset_topic != clear_topic:
            self.create_subscription(
                Empty, track_reset_topic, self.clear_interactions_callback, 10)

        # Record the canonical raw streams. Do not publish a second copy when
        # the RGB-D driver already uses these input topic names.
        rgb_record_topic = str(self.get_parameter('rgb_record_topic').value)
        depth_record_topic = str(self.get_parameter('depth_record_topic').value)
        self.rgb_raw_pub = (self.create_publisher(Image, rgb_record_topic, camera_qos)
                            if rgb_record_topic and rgb_record_topic != rgb_topic
                            else None)
        self.depth_raw_pub = (self.create_publisher(Image, depth_record_topic, camera_qos)
                              if depth_record_topic and depth_record_topic != depth_topic
                              else None)
        self.detections_pub = self.create_publisher(
            Detection2DArray, str(self.get_parameter('yolo_people_topic').value), 10)
        self.annotated_pub = self.create_publisher(
            Image, '/social_perception/annotated_image', 10)
        self.depth_visual_pub = self.create_publisher(
            Image, '/social_perception/depth_visualization', 10)
        self.markers_pub = self.create_publisher(
            MarkerArray, '/social_perception/person_markers', 10)
        self.people_pub = self.create_publisher(
            People, str(self.get_parameter('people_topic').value), 10)
        self.tracks_pub = self.create_publisher(
            People, str(self.get_parameter('people_tracks_topic').value), 10)
        self.observations_pub = self.create_publisher(
            PeopleObservations,
            str(self.get_parameter('people_observations_topic').value), 10)
        self.orientations_pub = self.create_publisher(
            PeopleOrientations,
            str(self.get_parameter('people_orientation_topic').value), 10)
        self.tracked_state_pub = self.create_publisher(
            String, str(self.get_parameter('tracked_state_topic').value), 10)
        # Latched, so anything that starts later still learns the pipeline is
        # up. It fires once YOLO has processed a real frame and the VLM has
        # finished loading, which is the moment people can actually be seen.
        self.ready_pub = self.create_publisher(
            Bool, '/social_perception/ready',
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.ready_lock = threading.Lock()
        self.ready_published = False
        self.first_frame_done = False
        # A disabled VLM is "loaded" by definition; a failed one must not hold
        # the signal back forever, so the worker reports either way.
        self.vlm_load_done = not self.vlm_enabled
        self.vlm_load_ok = not self.vlm_enabled

        # VLM output is deliberately not published here. A separate process
        # owns /social_perception/talking_interactions, preventing empty
        # heartbeat messages from Block B overwriting semantic VLM decisions.
        self.get_logger().info(
            f'Social perception ready: RGB={rgb_topic}, depth={depth_topic}, '
            f'color_info={info_topic}, depth_info={depth_info_topic or "disabled"}, '
            f'raw-record RGB={rgb_record_topic or "disabled"}, '
            f'depth={depth_record_topic or "disabled"}, '
            f'target={self.target_frame}, YOLO={yolo_path} on {self.yolo_device}; '
            'VLM interaction is a separate node.')

    def _declare_parameters(self):
        parameters = {
            'yolo_model_path': 'yolov8n.pt',
            'rgb_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            # Optional metadata from a separate depth camera. The depth image
            # must still be registered to the RGB image for pixel sampling.
            'depth_camera_info_topic': '/camera/depth/camera_info',
            # Canonical raw streams for recording. Both the simulated robot
            # camera and a real RGB-D driver may publish these directly.
            'rgb_record_topic': '/camera/color/image_raw',
            'depth_record_topic': '/camera/depth/image_raw',
            'yolo_confidence': 0.25,
            'yolo_image_size': 640,
            'yolo_device': 'cpu',
            'target_frame': 'map',
            'maximum_depth_age': 0.15,
            'minimum_depth': 0.2,
            'maximum_depth': 12.0,
            'bytetrack_config': 'config/bytetrack.yaml',
            'duplicate_merge_distance': 0.6,
            'tracking_timeout': 6.0,
            'velocity_smoothing': 0.35,
            'robot_frame': 'base_link',
            'odom_topic': '/odom',
            'maximum_odom_age': 0.2,
            'tracked_state_topic': '/people/tracked_state_json',
            'prediction_horizon_s': 2.0,
            'prediction_step_s': 0.5,
            'prediction_min_speed_mps': 0.10,
            'prediction_marker_diameter_m': 0.08,
            # COCO Pose keypoints below this score or without valid RGB-D depth
            # are kept out of the 3-D skeleton and body-yaw calculation.
            'keypoint_confidence': 0.35,
            'keypoint_depth_window_px': 5,
            'minimum_shoulder_width_m': 0.15,
            # Match the validated hip/shoulder/nose yaw estimator. History is
            # kept per ByteTrack id, never shared between different people.
            'orientation_keypoint_confidence': 0.50,
            # Hips/shoulders provide an undirected body axis. Face keypoints
            # select its forward sign so RViz never guesses front versus back.
            'orientation_nose_confidence': 0.60,
            'orientation_face_confidence': 0.40,
            'minimum_face_direction_m': 0.04,
            'minimum_face_alignment': 0.45,
            'orientation_flip_frames': 4,
            'orientation_hold_frames': -1,
            'maximum_shoulder_width_m': 0.60,
            'minimum_hip_width_m': 0.15,
            'maximum_hip_width_m': 0.50,
            'body_yaw_smoothing': 0.30,
            'orientation_marker_arrow_length_m': 0.45,
            'orientation_marker_lifetime_s': 7.0,
            'orientation_marker_arrow_color_r': 1.0,
            'orientation_marker_arrow_color_g': 0.0,
            'orientation_marker_arrow_color_b': 0.85,
            'people_topic': '/people',
            'people_tracks_topic': '/people/tracks',
            'people_observations_topic': '/people_observations',
            'people_orientation_topic': '/people/orientation',
            'yolo_people_topic': '/yolo/people',
            'interactions_topic': '/social_perception/talking_interactions',
            'clear_interactions_topic': '/animated_people/hide',
            # Gazebo's crossing actor teleports from B back to A each loop.
            # A real camera leaves this empty and keeps occlusion coasting.
            'track_reset_topic': '',
            # Legacy switch kept for backward-compatible parameter files. It
            # is force-disabled in __init__; social geometry is the only
            # active conversation proxy.
            'enable_vlm': False,
            'vlm_adapter_path': 'Saved_Model',
            'vlm_base_model': 'unsloth/Qwen2-VL-2B-Instruct-bnb-4bit',
            'vlm_load_in_4bit': True,
            'vlm_require_cuda': True,
            'vlm_max_new_tokens': 12,
            'vlm_force_answer_prefix': True,
            'vlm_min_pixels': 56 * 56,
            'vlm_max_pixels': 256 * 28 * 28,
            'vlm_inference_interval': 2.0,
            'vlm_refresh_interval': 15.0,
            'vlm_position_change_threshold': 0.25,
            'vlm_crop_margin': 0.05,
            'maximum_talking_distance': 3.0,
            'maximum_vlm_pairs': 1,
            'negative_answers_to_clear': 2,
            'interaction_timeout': 0.0,
            'log_vlm_results': True,
            'talking_prompt': (
                'Quan sát hai người trong ảnh. Họ có đang nói chuyện trực tiếp '
                'với nhau không? Chỉ trả lời đúng một JSON: '
                '{"talking":"có"} hoặc {"talking":"không"}.'),
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    @staticmethod
    def _resolve_file(value):
        candidate = Path(value).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend([
                Path.cwd() / candidate,
                Path(__file__).resolve().parent.parent / candidate,
                Path(__file__).resolve().parents[2] / candidate,
            ])
            try:
                candidates.append(Path(get_package_share_directory(
                    'social_perception')) / candidate)
            except Exception:  # Package index is unavailable before a build.
                pass
        for path in candidates:
            if path.exists():
                return path.resolve()
        return None

    def depth_callback(self, message):
        if self.depth_raw_pub is not None:
            self.depth_raw_pub.publish(message)
        with self.sensor_lock:
            self.depth_msg = message
            self.depth_buffer.append(message)
        self.depth_frames += 1

    def info_callback(self, message):
        self.camera_info = message

    def depth_info_callback(self, message):
        """Consume optional depth CameraInfo without treating it as RGB intrinsics.

        The node samples the depth image at RGB pixel locations, so only a
        depth stream registered to the colour camera is valid for this
        pipeline. Its CameraInfo is subscribed separately for a standard
        RGB-D topic layout and for downstream inspection.
        """
        self.depth_camera_info = message

    def odom_callback(self, message):
        """Keep the latest robot velocity for short-horizon relative prediction."""
        with self.sensor_lock:
            self.latest_odom = message
            self.odom_buffer.append(message)

    def nearest_sensor_message(self, buffer_name, header):
        """Return the buffered message nearest to a camera-frame timestamp."""
        target_stamp = stamp_seconds(header.stamp)
        with self.sensor_lock:
            messages = tuple(getattr(self, buffer_name))
        if not messages:
            return None, None
        message = min(
            messages,
            key=lambda item: abs(stamp_seconds(item.header.stamp) - target_stamp))
        age = abs(stamp_seconds(message.header.stamp) - target_stamp)
        return message, age

    def clear_interactions_callback(self, _message):
        """Immediately invalidate simulated people when they are hidden."""
        with self.state_lock:
            had_result = any(
                result['state'] != 'processing'
                for result in self.interaction_cache.values())
            self.latest_people = {}
            self.latest_observation_sequence += 1
            # Invalidate queued and in-flight work. CUDA generation cannot be
            # interrupted safely, but its answer must never enter the cache.
            self.scene_generation += 1
            self.interaction_cache.clear()
            self.vlm_inflight_pairs.clear()
            self.last_vlm_scenes.clear()
            self.last_logged_decisions.clear()
        self.discard_queued_vlm_work()
        if had_result:
            self.get_logger().info(
                'TRẠNG THÁI: KHÔNG CÓ CẶP NGƯỜI TRONG CAMERA')

    def mark_first_frame_done(self):
        with self.ready_lock:
            self.first_frame_done = True
        self.publish_ready_if_complete()

    def mark_vlm_load_done(self, loaded):
        with self.ready_lock:
            self.vlm_load_done = True
            self.vlm_load_ok = loaded
        self.publish_ready_if_complete()

    def publish_ready_if_complete(self):
        """Announce readiness once, after YOLO and the VLM have both settled."""
        with self.ready_lock:
            if self.ready_published or not (
                    self.first_frame_done and self.vlm_load_done):
                return
            self.ready_published = True
            vlm_ok = self.vlm_load_ok
        message = Bool()
        message.data = bool(vlm_ok)
        self.ready_pub.publish(message)
        if not self.vlm_enabled:
            state = 'đã tắt (enable_vlm=false)'
        elif vlm_ok:
            state = 'đã nạp xong'
        else:
            state = 'KHÔNG khả dụng, chỉ có định vị người'
        self.get_logger().info(f'SẴN SÀNG: camera + YOLO hoạt động, VLM {state}')

    def report_status(self):
        with self.state_lock:
            interaction_count = len(self.interaction_cache)
        self.get_logger().info(
            f'Input: RGB={self.rgb_frames}, depth={self.depth_frames}, '
            f'CameraInfo={"yes" if self.camera_info else "no"}, '
            f'cached interactions={interaction_count}')

    @staticmethod
    def rgb_array(message):
        channels = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4}.get(
            message.encoding)
        if channels is None:
            raise ValueError(f'unsupported RGB encoding: {message.encoding}')
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step)
        image = rows[:, :message.width * channels].reshape(
            message.height, message.width, channels)
        if message.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if message.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if message.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    @staticmethod
    def depth_array(message):
        if message.encoding == '32FC1':
            dtype, scale = np.dtype('>f4' if message.is_bigendian else '<f4'), 1.0
        elif message.encoding in ('16UC1', 'mono16'):
            dtype, scale = np.dtype('>u2' if message.is_bigendian else '<u2'), 0.001
        else:
            raise ValueError(f'unsupported depth encoding: {message.encoding}')
        row_items = message.step // dtype.itemsize
        rows = np.frombuffer(message.data, dtype=dtype).reshape(
            message.height, row_items)
        return rows[:, :message.width].astype(np.float32) * scale

    # Fraction of the bounding box sampled for distance: the torso, which is
    # the one part of a person that is neither background seen past an arm nor
    # floor seen past the feet.
    TORSO_SIDE, TORSO_TOP, TORSO_BOTTOM = 0.30, 0.20, 0.70
    # COCO human-pose indices used for a compact but useful RViz skeleton.
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
    LEFT_HIP, RIGHT_HIP = 11, 12
    COCO_SKELETON = (
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (0, 5), (0, 6),
    )
    @classmethod
    def torso_center(cls, box):
        """Pixel that median_depth's distance actually belongs to.

        A pinhole ray is only valid at the pixel its depth was measured at.
        Sampling the torso but casting through the feet placed every person
        several tens of centimetres towards the camera, because the two rays
        diverge once the camera is tilted.
        """
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5,
                y1 + 0.5 * (cls.TORSO_TOP + cls.TORSO_BOTTOM) * (y2 - y1))

    def median_depth(self, depth, box, rgb_shape):
        x1, y1, x2, y2 = box
        scale_x = depth.shape[1] / rgb_shape[1]
        scale_y = depth.shape[0] / rgb_shape[0]
        width, height = x2 - x1, y2 - y1
        rx1 = (x1 + self.TORSO_SIDE * width) * scale_x
        rx2 = (x2 - self.TORSO_SIDE * width) * scale_x
        ry1 = (y1 + self.TORSO_TOP * height) * scale_y
        ry2 = (y1 + self.TORSO_BOTTOM * height) * scale_y
        roi = depth[max(0, int(ry1)):min(depth.shape[0], int(ry2)),
                    max(0, int(rx1)):min(depth.shape[1], int(rx2))]
        valid = roi[np.isfinite(roi) & (roi >= self.min_depth) &
                    (roi <= self.max_depth)]
        return float(np.median(valid)) if valid.size >= 8 else None

    def keypoint_depth(self, depth, u, v, rgb_shape):
        """Median registered depth around one RGB keypoint pixel."""
        scale_x = depth.shape[1] / rgb_shape[1]
        scale_y = depth.shape[0] / rgb_shape[0]
        center_x, center_y = u * scale_x, v * scale_y
        radius_x = max(1, int(round(self.keypoint_depth_window * scale_x / 2.0)))
        radius_y = max(1, int(round(self.keypoint_depth_window * scale_y / 2.0)))
        left = max(0, int(round(center_x)) - radius_x)
        right = min(depth.shape[1], int(round(center_x)) + radius_x + 1)
        top = max(0, int(round(center_y)) - radius_y)
        bottom = min(depth.shape[0], int(round(center_y)) + radius_y + 1)
        roi = depth[top:bottom, left:right]
        valid = roi[np.isfinite(roi) & (roi >= self.min_depth) &
                    (roi <= self.max_depth)]
        return float(np.median(valid)) if valid.size >= 3 else None

    @staticmethod
    def result_keypoints(result, detection_index):
        """Return one YOLO Pose row as (x, y, confidence) triples.

        YOLO detection checkpoints expose ``keypoints=None``. Keeping that
        case explicit lets the RGB-D detector remain usable while clearly
        reporting that body orientation is unavailable.
        """
        keypoints = getattr(result, 'keypoints', None)
        if keypoints is None or getattr(keypoints, 'xy', None) is None:
            return None
        if detection_index >= len(keypoints.xy):
            return None
        xy = keypoints.xy[detection_index].cpu().tolist()
        confidence_data = getattr(keypoints, 'conf', None)
        confidence = (confidence_data[detection_index].cpu().tolist()
                      if confidence_data is not None else [1.0] * len(xy))
        return [(float(point[0]), float(point[1]), float(score))
                for point, score in zip(xy, confidence)]

    def localize_keypoints(self, pose_keypoints, depth, rgb_shape, transform):
        """Fill ROS keypoint messages and return valid 3-D points by COCO id."""
        keypoints_2d = []
        keypoints_3d = []
        points_3d = {}
        if pose_keypoints is None:
            return keypoints_2d, keypoints_3d, points_3d
        for keypoint_id, (u, v, confidence) in enumerate(pose_keypoints):
            point_2d = Keypoint2D()
            point_2d.id = keypoint_id
            point_2d.x, point_2d.y = u, v
            point_2d.confidence = confidence
            keypoints_2d.append(point_2d)
            if confidence < self.keypoint_confidence:
                continue
            depth_m = self.keypoint_depth(depth, u, v, rgb_shape)
            if depth_m is None or transform is None:
                continue
            point = self.point_in_target(u, v, depth_m, rgb_shape, transform)
            if point is None:
                continue
            point_3d = Keypoint3D()
            point_3d.id = keypoint_id
            point_3d.position.x, point_3d.position.y, point_3d.position.z = point
            point_3d.confidence = confidence
            keypoints_3d.append(point_3d)
            # Confidence travels with the point so orientation can apply its
            # stricter threshold independently from skeleton rendering.
            points_3d[keypoint_id] = (*point, confidence)
        return keypoints_2d, keypoints_3d, points_3d

    def body_yaw_from_keypoints(self, points_3d, previous_track):
        """Estimate a stable heading from body axis plus multi-point face cues.

        Hip and shoulder pairs only describe an *axis*: their two directions
        differ by 180 degrees and must not be averaged as directed headings.
        We therefore average them in double-angle space. The torso-to-nose
        vector and nose-to-eyes/ears vector vote for the forward half. A track
        must see an opposite vote for several frames before its heading flips;
        short face occlusions retain the last reliable heading briefly.
        """
        def confidence(point):
            return float(point[3]) if len(point) > 3 else 1.0

        def hold_previous_heading():
            if not previous_track or not previous_track.get('orientation_valid', False):
                return None, 0, 0
            held_frames = int(previous_track.get('orientation_hold_frames', 0)) + 1
            if (self.orientation_hold_frames >= 0 and
                    held_frames > self.orientation_hold_frames):
                return None, 0, 0
            return (float(previous_track['body_yaw_rad']),
                    int(previous_track.get('orientation_flip_votes', 0)),
                    held_frames)

        def valid_pair(first, second, minimum, maximum):
            if (first is None or second is None or
                    confidence(first) < self.orientation_keypoint_confidence or
                    confidence(second) < self.orientation_keypoint_confidence):
                return False
            return minimum <= math.hypot(second[0] - first[0], second[1] - first[1]) <= maximum

        def perpendicular_yaw(first, second):
            return wrap_angle(math.atan2(second[1] - first[1], second[0] - first[0]) +
                              math.pi / 2.0)

        axis_candidates = []
        left_hip = points_3d.get(self.LEFT_HIP)
        right_hip = points_3d.get(self.RIGHT_HIP)
        hips_valid = valid_pair(left_hip, right_hip,
                                self.minimum_hip_width, self.maximum_hip_width)
        if hips_valid:
            axis_candidates.append((perpendicular_yaw(left_hip, right_hip),
                                    2.0 * 0.5 * (confidence(left_hip) + confidence(right_hip))))

        left_shoulder = points_3d.get(self.LEFT_SHOULDER)
        right_shoulder = points_3d.get(self.RIGHT_SHOULDER)
        shoulders_valid = valid_pair(left_shoulder, right_shoulder,
                                     self.minimum_shoulder_width,
                                     self.maximum_shoulder_width)
        if shoulders_valid:
            axis_candidates.append((perpendicular_yaw(left_shoulder, right_shoulder),
                                    1.5 * 0.5 * (confidence(left_shoulder) +
                                                 confidence(right_shoulder))))

        nose = points_3d.get(self.NOSE)
        if shoulders_valid:
            torso_pair = (left_shoulder, right_shoulder)
        elif hips_valid:
            torso_pair = (left_hip, right_hip)
        else:
            torso_pair = None

        if (not axis_candidates or torso_pair is None or nose is None or
                confidence(nose) < self.orientation_nose_confidence):
            return hold_previous_heading()

        total_weight = sum(weight for _, weight in axis_candidates)
        if total_weight <= 0.0:
            return hold_previous_heading()
        axis_yaw = 0.5 * math.atan2(
            sum(weight * math.sin(2.0 * candidate)
                for candidate, weight in axis_candidates),
            sum(weight * math.cos(2.0 * candidate)
                for candidate, weight in axis_candidates))

        torso_center = (
            0.5 * (torso_pair[0][0] + torso_pair[1][0]),
            0.5 * (torso_pair[0][1] + torso_pair[1][1]),
        )
        face_cues = [(nose[0] - torso_center[0], nose[1] - torso_center[1], 1.0)]
        facial_points = [
            points_3d[keypoint_id] for keypoint_id in
            (self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR)
            if keypoint_id in points_3d and
            confidence(points_3d[keypoint_id]) >= self.orientation_face_confidence
        ]
        if len(facial_points) >= 2:
            facial_weight = sum(confidence(point) for point in facial_points)
            facial_center = (
                sum(confidence(point) * point[0] for point in facial_points) / facial_weight,
                sum(confidence(point) * point[1] for point in facial_points) / facial_weight,
            )
            face_cues.append((nose[0] - facial_center[0],
                              nose[1] - facial_center[1], 1.0))

        face_vote = 0.0
        face_vote_weight = 0.0
        for face_x, face_y, cue_weight in face_cues:
            face_distance = math.hypot(face_x, face_y)
            if face_distance < self.minimum_face_direction:
                continue
            alignment = ((math.cos(axis_yaw) * face_x +
                          math.sin(axis_yaw) * face_y) / face_distance)
            if abs(alignment) < self.minimum_face_alignment:
                continue
            face_vote += cue_weight * alignment
            face_vote_weight += cue_weight
        if face_vote_weight <= 0.0:
            return hold_previous_heading()
        face_vote /= face_vote_weight
        if abs(face_vote) < self.minimum_face_alignment:
            return hold_previous_heading()

        yaw = axis_yaw if face_vote > 0.0 else wrap_angle(axis_yaw + math.pi)
        flip_votes = 0
        if previous_track and previous_track.get('orientation_valid', False):
            previous_yaw = float(previous_track['body_yaw_rad'])
            if abs(wrap_angle(yaw - previous_yaw)) > math.pi / 2.0:
                flip_votes = int(previous_track.get('orientation_flip_votes', 0)) + 1
                if flip_votes < self.orientation_flip_frames:
                    return previous_yaw, flip_votes, 0
            else:
                flip_votes = 0
            yaw = wrap_angle(previous_yaw + self.body_yaw_smoothing *
                             wrap_angle(yaw - previous_yaw))
        return yaw, flip_votes, 0

    @staticmethod
    def set_person_yaw(person, yaw):
        qx, qy, qz, qw = yaw_quaternion(yaw)
        person.pose.orientation.x = qx
        person.pose.orientation.y = qy
        person.pose.orientation.z = qz
        person.pose.orientation.w = qw

    def camera_intrinsics(self, rgb_shape):
        """Return intrinsics scaled to the RGB image used by YOLO."""
        info = self.camera_info
        image_height, image_width = rgb_shape[:2]
        info_width = int(info.width) if info.width else image_width
        info_height = int(info.height) if info.height else image_height
        scale_x = image_width / max(1, info_width)
        scale_y = image_height / max(1, info_height)
        return (info.k[0] * scale_x, info.k[4] * scale_y,
                info.k[2] * scale_x, info.k[5] * scale_y)

    def target_transform(self, source_frame, stamp):
        """Look up one camera-to-target transform for the whole RGB-D frame."""
        if not source_frame:
            self.get_logger().warn(
                'Camera frame_id is empty; cannot transform RGB-D points',
                throttle_duration_sec=2.0)
            return None
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame, source_frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(
                f'TF {source_frame} -> {self.target_frame}: {error}',
                throttle_duration_sec=2.0)
            return None

    @staticmethod
    def transform_optical_point(optical, transform):
        rotated = rotate_vector(optical, transform.transform.rotation)
        translation = transform.transform.translation
        return (rotated[0] + translation.x,
                rotated[1] + translation.y,
                rotated[2] + translation.z)

    def point_in_target(self, u, v, depth, rgb_shape, transform):
        fx, fy, cx, cy = self.camera_intrinsics(rgb_shape)
        if fx <= 0.0 or fy <= 0.0:
            return None
        optical = ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)
        return self.transform_optical_point(optical, transform)

    def track_velocity(self, track_id, point, stamp_value):
        """Estimate map-frame velocity for the stable ID assigned by ByteTrack."""
        previous = self.tracks.get(track_id)
        if previous is None:
            return (0.0, 0.0, 0.0)
        dt = stamp_value - previous['stamp']
        if dt <= 1e-3:
            return (0.0, 0.0, 0.0)
        measured = tuple((point[index] - previous['point'][index]) / dt
                         for index in range(3))
        return tuple(
            self.velocity_alpha * measured[index] +
            (1.0 - self.velocity_alpha) * previous['velocity'][index]
            for index in range(3))

    def reset_bytetrack(self):
        """Forget tracks when the simulator removes all actors."""
        predictor = getattr(self.yolo, 'predictor', None)
        for tracker in getattr(predictor, 'trackers', []):
            tracker.reset()

    def robot_transform(self, header):
        """Return the map/target-frame to robot-frame transform at an RGB stamp."""
        try:
            return self.tf_buffer.lookup_transform(
                self.robot_frame, self.target_frame,
                rclpy.time.Time.from_msg(header.stamp),
                timeout=Duration(seconds=0.05))
        except TransformException as error:
            self.get_logger().warn(
                f'TF {self.target_frame} -> {self.robot_frame}: {error}',
                throttle_duration_sec=2.0)
            return None

    def robot_velocity(self, header):
        """Return the current robot linear velocity in robot-frame coordinates."""
        odom, age = self.nearest_sensor_message('odom_buffer', header)
        if odom is None:
            return (0.0, 0.0, 0.0)
        if age > self.max_odom_age:
            self.get_logger().warn(
                f'Odom is {age:.3f}s away from the RGB frame; using zero robot velocity',
                throttle_duration_sec=2.0)
            return (0.0, 0.0, 0.0)
        velocity = odom.twist.twist.linear
        return (float(velocity.x), float(velocity.y), float(velocity.z))

    @staticmethod
    def relative_position(x, y):
        degrees = math.degrees(math.atan2(y, x))
        if -30.0 <= degrees <= 30.0:
            return 'front'
        if 30.0 < degrees < 150.0:
            return 'front_left' if degrees < 90.0 else 'rear_left'
        if -150.0 < degrees < -30.0:
            return 'front_right' if degrees > -90.0 else 'rear_right'
        return 'rear'

    def motion_state(self, x, y, velocity_x, velocity_y):
        speed = math.hypot(velocity_x, velocity_y)
        if speed < self.prediction_min_speed:
            return 'stationary', 'stationary', speed
        distance = math.hypot(x, y)
        radial_speed = 0.0 if distance < 1e-3 else (
            x * velocity_x + y * velocity_y) / distance
        if radial_speed <= -0.5 * self.prediction_min_speed:
            direction = 'toward_robot'
        elif radial_speed >= 0.5 * self.prediction_min_speed:
            direction = 'away_from_robot'
        elif velocity_y <= -0.5 * self.prediction_min_speed:
            direction = 'left_to_right'
        elif velocity_y >= 0.5 * self.prediction_min_speed:
            direction = 'right_to_left'
        elif velocity_x > 0.0:
            direction = 'forward'
        else:
            direction = 'backward'
        return 'moving', direction, speed

    def build_tracked_state(self, header, visible_tracks):
        """Build the map-frame Block-B JSON contract from visible tracks."""
        people = []
        for track_id, track in sorted(visible_tracks.items()):
            position = track['point']
            velocity = track['velocity']
            speed = math.hypot(velocity[0], velocity[1])
            heading_valid = speed >= self.prediction_min_speed
            heading = math.atan2(velocity[1], velocity[0]) if heading_valid else 0.0

            def trajectory_point(dt):
                # Block B publishes only the constant-velocity position at dt.
                # Prediction uncertainty is intentionally not part of this
                # contract, so downstream Block D keeps its configured social
                # Gaussian dimensions instead of uncertainty-based expansion.
                return {
                    'dt_s': round(dt, 3),
                    'x': round(position[0] + velocity[0] * dt, 3),
                    'y': round(position[1] + velocity[1] * dt, 3),
                }

            trajectory = []
            dt = self.prediction_step
            while dt < self.prediction_horizon - 1e-6:
                trajectory.append(trajectory_point(dt))
                dt += self.prediction_step
            trajectory.append(trajectory_point(self.prediction_horizon))
            people.append({
                'track_id': track_id,
                'position_m': {
                    'x': round(position[0], 3), 'y': round(position[1], 3),
                },
                'velocity_mps': {
                    'vx': round(velocity[0], 3), 'vy': round(velocity[1], 3),
                },
                'motion_heading_rad': round(heading, 3),
                'motion_heading_valid': heading_valid,
                # Body-facing orientation is estimated from 3-D shoulder/hip
                # axis and face keypoints, independently of motion heading.
                'body_orientation_rad': round(
                    float(track.get('body_yaw_rad', 0.0)), 3),
                'body_orientation_valid': bool(
                    track.get('orientation_valid', False)),
                'prediction': {
                    'horizon_s': round(self.prediction_horizon, 3),
                    'trajectory_m': trajectory,
                },
            })
        return {
            'header': {
                'stamp_ns': (
                    int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)),
                'frame_id': self.target_frame,
            },
            'people': people,
        }

    def publish_tracked_state(self, header, visible_tracks):
        state = self.build_tracked_state(header, visible_tracks)
        if state is None:
            return
        message = String()
        message.data = json.dumps(state, ensure_ascii=False, separators=(',', ':'))
        self.tracked_state_pub.publish(message)

    def rgb_callback(self, rgb_msg):
        if self.rgb_raw_pub is not None:
            self.rgb_raw_pub.publish(rgb_msg)
        self.rgb_frames += 1
        # A hide/clear callback may run while YOLO is processing this frame.
        # Capturing the generation here lets us reject that stale frame later.
        reset_bytetrack = False
        with self.state_lock:
            frame_generation = self.scene_generation
            if self.tracks_generation != frame_generation:
                # A hide event happened between frames. Reset here, on the RGB
                # callback group that owns tracking, so the clear callback can
                # never reset ByteTrack while association is running it.
                self.tracks = {}
                self.tracks_generation = frame_generation
                reset_bytetrack = True
        # Keep one immutable reference throughout this inference even if the
        # depth callback receives a newer frame on the second executor thread.
        depth_msg, depth_age = self.nearest_sensor_message('depth_buffer', rgb_msg.header)
        if depth_msg is None or self.camera_info is None:
            self.get_logger().warn('Waiting for depth image and CameraInfo',
                                   throttle_duration_sec=2.0)
            return
        if depth_age > self.max_depth_age:
            self.get_logger().warn(
                'RGB and depth timestamps are not synchronized: '
                f'nearest delta={depth_age:.3f}s',
                                   throttle_duration_sec=2.0)
            return
        try:
            image = self.rgb_array(rgb_msg)
            depth_image = self.depth_array(depth_msg)
        except (ValueError, TypeError) as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)
            return

        self.publish_depth_visualization(depth_image, depth_msg.header)
        yolo_uses_cuda = self.yolo_device.lower() not in ('cpu', 'mps')
        lock_context = (self.ml_execution_lock if yolo_uses_cuda
                        else contextlib.nullcontext())
        with lock_context:
            if reset_bytetrack:
                self.reset_bytetrack()
            result = self.yolo.track(
                source=image, classes=[0], conf=self.confidence,
                imgsz=self.image_size, device=self.yolo_device,
                tracker=str(self.bytetrack_config), persist=True,
                verbose=False)[0]
        detections = Detection2DArray()
        detections.header = rgb_msg.header
        people = People()
        people.header.stamp = rgb_msg.header.stamp
        people.header.frame_id = self.target_frame
        observations = PeopleObservations()
        observations.header = rgb_msg.header
        orientations = PeopleOrientations()
        orientations.header = people.header
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        annotated = image.copy()
        candidates = []
        new_tracks = {}
        visible_tracks = {}
        stamp_value = stamp_seconds(rgb_msg.header.stamp)

        source_frame = (self.camera_info.header.frame_id or
                        rgb_msg.header.frame_id or depth_msg.header.frame_id)
        transform = None
        if result.boxes is not None and len(result.boxes) > 0:
            transform = self.target_transform(source_frame, rgb_msg.header.stamp)

        if result.boxes is not None:
            for detection_index, box in enumerate(result.boxes):
                bbox = tuple(float(value) for value in box.xyxy[0].cpu().tolist())
                score = float(box.conf[0].cpu())
                detection = self.make_detection(rgb_msg, bbox, score)
                detections.detections.append(detection)
                z = self.median_depth(depth_image, bbox, image.shape)
                x1, y1, x2, y2 = bbox
                point = None if z is None or transform is None else self.point_in_target(
                    *self.torso_center(bbox), z, image.shape, transform)
                color = (0, 200, 255) if point is not None else (0, 0, 255)
                # ByteTrack assigns this numeric ID. Show that identity on the
                # camera image so an operator can follow one track across
                # frames; ``person_<id>`` remains the ROS topic contract below.
                track_id = None if box.id is None else int(box.id[0].item())
                track_label = f'ID {track_id}' if track_id is not None else 'ID ?'
                # Detection confidence remains in PersonObservation for logic
                # and debugging, but the camera overlay is intentionally
                # limited to the track identity and measured depth.
                label = track_label + (
                    f' {z:.1f}m' if z is not None else ' no-depth')
                cv2.rectangle(annotated, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), max(20, int(y1) - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                pose_keypoints = self.result_keypoints(result, detection_index)
                if pose_keypoints is None and not self.pose_model_warning_logged:
                    self.get_logger().warn(
                        'YOLO checkpoint không xuất keypoints; skeleton/body-yaw '
                        'sẽ không có. Hãy dùng YOLO Pose checkpoint.')
                    self.pose_model_warning_logged = True
                self.draw_pose_overlay(annotated, pose_keypoints)
                if point is None:
                    continue

                if track_id is None:
                    continue
                previous_track = self.tracks.get(track_id)
                velocity = self.track_velocity(track_id, point, stamp_value)
                keypoints_2d, keypoints_3d, points_3d = self.localize_keypoints(
                    pose_keypoints, depth_image, image.shape, transform)
                body_yaw, orientation_flip_votes, orientation_hold_frames = (
                    self.body_yaw_from_keypoints(points_3d, previous_track))
                orientation_valid = body_yaw is not None
                new_tracks[track_id] = {
                    'point': point,
                    'velocity': velocity,
                    'stamp': stamp_value,
                    'orientation_valid': orientation_valid,
                    'body_yaw_rad': 0.0 if body_yaw is None else body_yaw,
                    'orientation_flip_votes': orientation_flip_votes,
                    'orientation_hold_frames': orientation_hold_frames,
                }
                visible_tracks[track_id] = new_tracks[track_id]
                # Keep downstream IDs stable: VLM pairs observations with
                # ``People`` by this exact string, while the overlay uses the
                # raw ByteTrack integer above the bounding box.
                detection.id = f'person_{track_id}'
                person = Person()
                person.id = detection.id
                person.pose.position.x, person.pose.position.y, person.pose.position.z = point
                person.pose.orientation.w = 1.0
                if orientation_valid:
                    self.set_person_yaw(person, body_yaw)
                (person.velocity.linear.x, person.velocity.linear.y,
                 person.velocity.linear.z) = velocity
                people.people.append(person)
                observation = PersonObservation()
                observation.id = person.id
                observation.confidence = score
                (observation.bbox_x_min, observation.bbox_y_min,
                 observation.bbox_x_max, observation.bbox_y_max) = bbox
                observation.distance_m = float(z)
                observation.orientation_valid = orientation_valid
                observation.body_yaw_rad = 0.0 if body_yaw is None else body_yaw
                observation.keypoints_2d = keypoints_2d
                observation.keypoints_3d = keypoints_3d
                observations.people.append(observation)
                orientation = PersonOrientation()
                orientation.id = person.id
                orientation.valid = orientation_valid
                orientation.body_yaw_rad = observation.body_yaw_rad
                orientations.people.append(orientation)
                candidates.append({'person': person, 'bbox': bbox})
                markers.markers.extend(self.person_markers(
                    people.header, person, track_id, points_3d, body_yaw))
                markers.markers.extend(self.predicted_trajectory_markers(
                    people.header, track_id, new_tracks[track_id]))
                if orientation_valid:
                    cv2.putText(
                        annotated, f'yaw {math.degrees(body_yaw):+.0f} deg',
                        (int(x1), min(image.shape[0] - 28, int(y2) + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # A person the robot is standing in front of is still a person.
        #
        # Retaining the track used to protect only the id counter: the occluded
        # person still dropped out of /people. The robot approaching to
        # pass between two people is itself what can briefly occlude them.
        #
        # Republishing the track at its last known position keeps the pair
        # intact, and keeps the hidden person in the costmap while the robot
        # cannot see them, which is the conservative reading either way.
        published_points = [(person.pose.position.x, person.pose.position.y)
                            for person in people.people]
        for track_id, state in self.tracks.items():
            age = stamp_value - state['stamp']
            if track_id in new_tracks:
                continue
            if not 0.0 <= age <= self.track_timeout:
                continue
            # A track sitting on top of somebody already published this frame
            # is a second id for that same person, left over from a frame where
            # association picked the other one. Coasting it would put one
            # person in /people twice, and the closest-pair search would then
            # hand the VLM a person paired with themselves instead of the two
            # people actually talking. Let it lapse.
            if any(math.hypot(state['point'][0] - x, state['point'][1] - y)
                   < self.duplicate_distance for x, y in published_points):
                continue
            new_tracks[track_id] = state
            published_points.append((state['point'][0], state['point'][1]))
            coasted = Person()
            coasted.id = f'person_{track_id}'
            (coasted.pose.position.x, coasted.pose.position.y,
             coasted.pose.position.z) = state['point']
            coasted.pose.orientation.w = 1.0
            orientation_valid = bool(state.get('orientation_valid', False))
            body_yaw = float(state.get('body_yaw_rad', 0.0))
            if orientation_valid:
                self.set_person_yaw(coasted, body_yaw)
            (coasted.velocity.linear.x, coasted.velocity.linear.y,
             coasted.velocity.linear.z) = state['velocity']
            people.people.append(coasted)
            orientation = PersonOrientation()
            orientation.id = coasted.id
            orientation.valid = orientation_valid
            orientation.body_yaw_rad = body_yaw
            orientations.people.append(orientation)
            markers.markers.extend(self.person_markers(
                people.header, coasted, track_id, {},
                body_yaw if orientation_valid else None))
            markers.markers.extend(self.predicted_trajectory_markers(
                people.header, track_id, state))
        self.tracks = new_tracks
        visible_ids = {item.id for item in people.people}
        removed_completed_result = False
        with self.state_lock:
            if frame_generation != self.scene_generation:
                # This RGB message started before a hide/clear event. Publishing
                # it would make old actors visible again and could validate an
                # obsolete VLM answer.
                self.tracks = {}
                self.tracks_generation = self.scene_generation
                return
            # Somebody leaving the frame invalidates work about *them*, not the
            # whole scene.
            #
            # This used to bump scene_generation, which every queued and running
            # inference is tagged with, so one person dropping out threw away an
            # in-flight answer about a different pair entirely. Tracking loses a
            # person routinely -- YOLO runs at ~6.6 fps on the CPU and a walking
            # person is missed for a few frames at a time -- so with people
            # moving, inferences were being discarded and restarted faster than
            # they could finish, and no result ever reached the topic.
            #
            # Dropping the per-pair markers is enough: the worker re-checks
            # membership before inferring, and pair_visible_after re-checks it
            # again before publishing. Clearing the markers here also lets an
            # affected pair be re-enqueued immediately instead of waiting out
            # vlm_refresh_interval.
            vanished_ids = set(self.latest_people) - visible_ids
            dropped_inflight = []
            if vanished_ids:
                for registry in (self.vlm_inflight_pairs, self.last_vlm_scenes):
                    for pair in [pair for pair in registry
                                 if not vanished_ids.isdisjoint(pair)]:
                        if registry is self.vlm_inflight_pairs:
                            dropped_inflight.append(pair)
                        registry.pop(pair, None)
            self.latest_people = {item.id: item for item in people.people}
            self.latest_header = people.header
            self.latest_observation_sequence += 1
            stale_pairs = [
                pair for pair in self.interaction_cache
                if not all(member_id in visible_ids for member_id in pair)
            ]
            for pair in stale_pairs:
                removed = self.interaction_cache.pop(pair)
                if removed['state'] != 'processing':
                    removed_completed_result = True
                self.last_logged_decisions.pop(pair, None)
        if dropped_inflight:
            # If this appears repeatedly while two people are talking, tracking
            # is losing them faster than the VLM can answer: raise
            # tracking_timeout or ByteTrack's track_buffer rather than blaming
            # the model.
            self.get_logger().warn(
                'MẤT DẤU giữa lúc suy luận, bỏ kết quả VLM của '
                f'{dropped_inflight} (biến mất: {sorted(vanished_ids)})',
                throttle_duration_sec=5.0)
        if removed_completed_result and len(visible_ids) < 2:
            self.get_logger().info(
                'TRẠNG THÁI: KHÔNG CÓ CẶP NGƯỜI TRONG CAMERA')
        self.mark_first_frame_done()
        self.detections_pub.publish(detections)
        self.people_pub.publish(people)
        self.tracks_pub.publish(people)
        self.observations_pub.publish(observations)
        self.orientations_pub.publish(orientations)
        self.publish_tracked_state(rgb_msg.header, visible_tracks)
        self.markers_pub.publish(markers)
        self.annotated_pub.publish(self.cv_image_message(annotated, rgb_msg.header))
        # VLM pair-crop generation and talking/not-talking inference are
        # disabled.
        # self.enqueue_vlm_scene(image, candidates, people.header)

    @staticmethod
    def make_detection(rgb_msg, bbox, score):
        x1, y1, x2, y2 = bbox
        detection = Detection2D()
        detection.header = rgb_msg.header
        detection.bbox.center.position.x = (x1 + x2) * 0.5
        detection.bbox.center.position.y = (y1 + y2) * 0.5
        detection.bbox.size_x = x2 - x1
        detection.bbox.size_y = y2 - y1
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = 'person'
        hypothesis.hypothesis.score = score
        detection.results.append(hypothesis)
        return detection

    def person_markers(self, header, person, track_id, keypoints_3d, body_yaw):
        """Create RViz body, skeleton, and optional forward-yaw markers."""
        marker = Marker()
        marker.header = header
        marker.ns = 'rgbd_people'
        marker.id = track_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        # A message field assignment stores the object itself, so the raised
        # marker height below would otherwise be written back into the Person
        # already queued for /people.
        marker.pose = copy.deepcopy(person.pose)
        marker.pose.position.z = max(0.9, person.pose.position.z)
        marker.scale.x, marker.scale.y, marker.scale.z = 0.55, 0.55, 1.8
        marker.color.r, marker.color.g = 0.0, 1.0
        marker.color.b, marker.color.a = 0.1, 0.55
        marker.lifetime = Duration(seconds=0.4).to_msg()
        markers = [marker]

        if keypoints_3d:
            joints = Marker()
            joints.header = header
            joints.ns = 'pose_keypoints'
            joints.id = track_id
            joints.type = Marker.SPHERE_LIST
            joints.action = Marker.ADD
            joints.scale.x = joints.scale.y = joints.scale.z = 0.08
            joints.color.r, joints.color.g, joints.color.b, joints.color.a = 0.1, 0.8, 1.0, 0.95
            joints.points = [Point(x=point[0], y=point[1], z=point[2])
                             for point in keypoints_3d.values()]
            joints.lifetime = Duration(seconds=0.4).to_msg()
            markers.append(joints)

            skeleton = Marker()
            skeleton.header = header
            skeleton.ns = 'pose_skeleton'
            skeleton.id = track_id
            skeleton.type = Marker.LINE_LIST
            skeleton.action = Marker.ADD
            skeleton.scale.x = 0.035
            skeleton.color.r, skeleton.color.g, skeleton.color.b, skeleton.color.a = 1.0, 0.75, 0.0, 0.95
            for first, second in SocialVlmPerception.COCO_SKELETON:
                if first in keypoints_3d and second in keypoints_3d:
                    first_point, second_point = keypoints_3d[first], keypoints_3d[second]
                    skeleton.points.extend((
                        Point(x=first_point[0], y=first_point[1], z=first_point[2]),
                        Point(x=second_point[0], y=second_point[1], z=second_point[2]),
                    ))
            if skeleton.points:
                skeleton.lifetime = Duration(seconds=0.4).to_msg()
                markers.append(skeleton)

        if body_yaw is not None:
            arrow = Marker()
            arrow.header = header
            arrow.ns = 'body_yaw'
            arrow.id = track_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = copy.deepcopy(person.pose)
            arrow.pose.position.z = max(1.2, person.pose.position.z + 0.35)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = (
                self.orientation_arrow_length, 0.12, 0.18)
            arrow.color.r, arrow.color.g, arrow.color.b = self.orientation_arrow_color
            arrow.color.a = 0.95
            arrow.lifetime = Duration(seconds=self.orientation_arrow_lifetime).to_msg()
            markers.append(arrow)
        return markers

    def predicted_trajectory_markers(self, header, track_id, track):
        """Draw the map-frame constant-velocity prediction used by Block-B."""
        velocity_x, velocity_y, _ = track['velocity']
        if math.hypot(velocity_x, velocity_y) < self.prediction_min_speed:
            return []

        point = track['point']
        # Keep the visual marker in target_frame (normally map).  Block-B also
        # exports robot-relative coordinates, but putting those directly into a
        # map-frame marker would make the trajectory shift whenever the robot
        # moves.  The tracker velocity and point are already map-frame values.
        trajectory = [Point(x=point[0], y=point[1], z=point[2])]
        future_points = []
        dt = self.prediction_step
        while dt < self.prediction_horizon - 1e-6:
            future_points.append(Point(
                x=point[0] + velocity_x * dt,
                y=point[1] + velocity_y * dt,
                z=point[2]))
            dt += self.prediction_step
        future_points.append(Point(
            x=point[0] + velocity_x * self.prediction_horizon,
            y=point[1] + velocity_y * self.prediction_horizon,
            z=point[2]))
        trajectory.extend(future_points)

        line = Marker()
        line.header = header
        line.ns = 'predicted_trajectory_line'
        line.id = track_id
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.045
        line.color.r, line.color.g, line.color.b, line.color.a = 0.0, 0.9, 1.0, 0.95
        line.points = trajectory
        line.lifetime = Duration(seconds=0.4).to_msg()

        dots = Marker()
        dots.header = header
        dots.ns = 'predicted_trajectory_points'
        dots.id = track_id
        dots.type = Marker.SPHERE_LIST
        dots.action = Marker.ADD
        dots.pose.orientation.w = 1.0
        dots.scale.x = dots.scale.y = dots.scale.z = self.prediction_marker_diameter
        dots.color.r, dots.color.g, dots.color.b, dots.color.a = 1.0, 0.0, 0.0, 0.95
        # The actor/body marker is the present position.  Dots are only the
        # predicted positions at prediction_step through prediction_horizon.
        dots.points = future_points
        dots.lifetime = Duration(seconds=0.4).to_msg()
        return [line, dots]

    def draw_pose_overlay(self, image, pose_keypoints):
        """Draw YOLO Pose joints in the annotated RGB image for quick checks."""
        if pose_keypoints is None:
            return
        visible = {
            keypoint_id: (int(round(u)), int(round(v)))
            for keypoint_id, (u, v, confidence) in enumerate(pose_keypoints)
            if confidence >= self.keypoint_confidence
        }
        for first, second in self.COCO_SKELETON:
            if first in visible and second in visible:
                cv2.line(image, visible[first], visible[second], (0, 215, 255), 2,
                         lineType=cv2.LINE_AA)
        for point in visible.values():
            cv2.circle(image, point, 4, (0, 255, 80), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, point, 5, (20, 20, 20), 1, lineType=cv2.LINE_AA)

    @staticmethod
    def cv_image_message(image, header):
        output = Image()
        output.header = header
        output.height, output.width = image.shape[:2]
        output.encoding = 'bgr8'
        output.is_bigendian = 0
        output.step = output.width * 3
        output.data = image.tobytes()
        return output

    def publish_depth_visualization(self, depth_image, depth_header):
        finite = np.isfinite(depth_image) & (depth_image >= self.min_depth) & (
            depth_image <= self.max_depth)
        normalized = np.zeros(depth_image.shape, dtype=np.uint8)
        if np.any(finite):
            near = float(np.percentile(depth_image[finite], 2.0))
            far = float(np.percentile(depth_image[finite], 98.0))
            normalized[finite] = np.clip(
                255.0 * (depth_image[finite] - near) / max(0.1, far - near),
                0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        self.depth_visual_pub.publish(
            self.cv_image_message(depth_color, depth_header))

    def enqueue_vlm_scene(self, image, candidates, observation_header):
        """LEGACY / DISABLED: queue a VLM pair crop if the branch is restored."""
        if not self.vlm_enabled or len(candidates) < 2:
            self.discard_queued_vlm_work()
            return
        now = time.monotonic()
        if now - self.last_vlm_enqueue < self.vlm_interval:
            return
        pair_candidates = []
        for first_index in range(len(candidates)):
            for second_index in range(first_index + 1, len(candidates)):
                first, second = candidates[first_index], candidates[second_index]
                distance = math.hypot(
                    first['person'].pose.position.x - second['person'].pose.position.x,
                    first['person'].pose.position.y - second['person'].pose.position.y)
                if distance <= self.max_pair_distance:
                    pair_candidates.append((distance, first, second))
        pair_candidates.sort(key=lambda item: item[0])
        with self.state_lock:
            observation_sequence = self.latest_observation_sequence
            scene_generation = self.scene_generation
        work = []
        # Walk every pair and stop once enough work is collected, rather than
        # only ever looking at the closest `max_vlm_pairs`. Truncating first
        # meant the nearest pair permanently occupied the only slot: while its
        # decision was still fresh the loop skipped it and ended, so a second
        # conversation further away was never once put to the model for its
        # separate diagnostic interaction result.
        for _, first, second in pair_candidates:
            if len(work) >= self.max_vlm_pairs:
                break
            pair = tuple(sorted((first['person'].id, second['person'].id)))
            positions = {
                first['person'].id: (
                    first['person'].pose.position.x,
                    first['person'].pose.position.y),
                second['person'].id: (
                    second['person'].pose.position.x,
                    second['person'].pose.position.y),
            }
            with self.state_lock:
                previous = self.last_vlm_scenes.get(pair)
                in_flight = pair in self.vlm_inflight_pairs
            moved = previous is None
            if previous is not None:
                moved = any(math.hypot(
                    positions[member_id][0] - previous['positions'][member_id][0],
                    positions[member_id][1] - previous['positions'][member_id][1],
                ) >= self.vlm_position_threshold for member_id in pair)
            refresh_due = (previous is None or
                           now - previous['time'] >= self.vlm_refresh_interval)
            if in_flight or not (moved or refresh_due):
                continue
            crop = self.pair_crop(image, first['bbox'], second['bbox'])
            work.append({
                'first_id': first['person'].id,
                'second_id': second['person'].id,
                'people': [copy.deepcopy(first['person']),
                           copy.deepcopy(second['person'])],
                'observation_header': copy.deepcopy(observation_header),
                'observation_sequence': observation_sequence,
                'scene_generation': scene_generation,
                'positions': positions,
                'crop': crop,
            })
        if not work:
            return
        self.discard_queued_vlm_work()
        with self.state_lock:
            if scene_generation != self.scene_generation:
                return
            for item in work:
                pair = tuple(sorted((item['first_id'], item['second_id'])))
                self.vlm_inflight_pairs[pair] = scene_generation
                self.last_vlm_scenes[pair] = {
                    'positions': item['positions'],
                    'time': now,
                    'scene_generation': scene_generation,
                }
        try:
            self.work_queue.put_nowait(work)
        except queue.Full:
            with self.state_lock:
                for item in work:
                    pair = tuple(sorted((item['first_id'], item['second_id'])))
                    if self.vlm_inflight_pairs.get(pair) == scene_generation:
                        self.vlm_inflight_pairs.pop(pair, None)
            return
        self.last_vlm_enqueue = now

    def pair_visible_after(self, pair, sequence, scene_generation, timeout=5.0):
        """Validate an inference against a camera frame captured after it."""
        deadline = time.monotonic() + timeout
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            with self.state_lock:
                current_sequence = self.latest_observation_sequence
                visible_ids = set(self.latest_people)
                current_generation = self.scene_generation
            if current_generation != scene_generation:
                return False
            if current_sequence > sequence:
                return all(member_id in visible_ids for member_id in pair)
            self.stop_event.wait(0.05)
        return False

    def discard_queued_vlm_work(self):
        """Drop camera crops that are no longer the newest observation."""
        discarded_pairs = []
        while True:
            try:
                work = self.work_queue.get_nowait()
                discarded_pairs.extend(
                    (tuple(sorted((item['first_id'], item['second_id']))),
                     item['scene_generation']) for item in work)
            except queue.Empty:
                break
        if discarded_pairs:
            with self.state_lock:
                for pair, generation in discarded_pairs:
                    if self.vlm_inflight_pairs.get(pair) == generation:
                        self.vlm_inflight_pairs.pop(pair, None)

    def pair_crop(self, image, first_box, second_box):
        height, width = image.shape[:2]
        x1 = min(first_box[0], second_box[0])
        y1 = min(first_box[1], second_box[1])
        x2 = max(first_box[2], second_box[2])
        y2 = max(first_box[3], second_box[3])
        margin_x = (x2 - x1) * self.crop_margin
        margin_y = (y2 - y1) * self.crop_margin
        left, top = max(0, int(x1 - margin_x)), max(0, int(y1 - margin_y))
        right, bottom = min(width, int(x2 + margin_x)), min(height, int(y2 + margin_y))
        return image[top:bottom, left:right].copy()

    def finish_vlm_item(self, pair, scene_generation, keep_scene):
        """Release one queued/running marker without touching newer work."""
        with self.state_lock:
            if self.vlm_inflight_pairs.get(pair) == scene_generation:
                self.vlm_inflight_pairs.pop(pair, None)
            scene = self.last_vlm_scenes.get(pair)
            if scene is not None and scene['scene_generation'] == scene_generation:
                if keep_scene:
                    scene['time'] = time.monotonic()
                else:
                    self.last_vlm_scenes.pop(pair, None)

    def vlm_worker(self):
        """LEGACY / DISABLED: load and run the Qwen talking classifier."""
        adapter_value = str(self.get_parameter('vlm_adapter_path').value)
        adapter_path = self._resolve_file(adapter_value)
        if adapter_path is None or not (adapter_path / 'adapter_config.json').is_file():
            self.get_logger().error(
                f'VLM adapter not found at {adapter_value}; people localization remains active')
            self.mark_vlm_load_done(False)
            return
        try:
            backend = VlmBackend(
                adapter_path,
                str(self.get_parameter('vlm_base_model').value),
                bool(self.get_parameter('vlm_load_in_4bit').value),
                bool(self.get_parameter('vlm_require_cuda').value),
                int(self.get_parameter('vlm_max_new_tokens').value),
                int(self.get_parameter('vlm_min_pixels').value),
                int(self.get_parameter('vlm_max_pixels').value),
                self.get_logger(),
                self.ml_execution_lock,
                bool(self.get_parameter('vlm_force_answer_prefix').value))
        except Exception as error:  # Keep YOLO/depth available when ML deps are absent.
            self.get_logger().error(
                f'Unable to load VLM ({type(error).__name__}: {error}); '
                'people localization remains active')
            self.mark_vlm_load_done(False)
            return
        self.mark_vlm_load_done(True)

        prompt = str(self.get_parameter('talking_prompt').value)
        while not self.stop_event.is_set():
            try:
                work = self.work_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            for item in work:
                first_id = item['first_id']
                second_id = item['second_id']
                crop = item['crop']
                pair = tuple(sorted((first_id, second_id)))
                item_generation = item['scene_generation']
                with self.state_lock:
                    current_sequence = self.latest_observation_sequence
                    visible_ids = set(self.latest_people)
                    current_generation = self.scene_generation
                if (current_generation != item_generation or
                        (current_sequence > item['observation_sequence'] and
                        not all(member_id in visible_ids for member_id in pair))):
                    self.finish_vlm_item(pair, item_generation, keep_scene=False)
                    continue
                # Publish an explicit in-progress record immediately. This
                # prevents an empty topic from looking like a stalled node
                # during a long Qwen generation call.
                started_stamp = self.get_clock().now().to_msg()
                with self.state_lock:
                    start_is_current = (
                        self.scene_generation == item_generation and
                        all(member_id in self.latest_people for member_id in pair))
                    # Keep the last completed answer visible while refreshing
                    # it. Only publish "processing" before the first answer.
                    if start_is_current and pair not in self.interaction_cache:
                        self.interaction_cache[pair] = {
                            'state': 'processing',
                            'talking': False,
                            'confidence': 0.0,
                            'response': 'VLM inference in progress',
                            'people': item['people'],
                            'observation_header': item['observation_header'],
                            'inference_stamp': started_stamp,
                            'time': time.monotonic(),
                            'scene_generation': item_generation,
                        }
                if not start_is_current:
                    self.finish_vlm_item(pair, item_generation, keep_scene=False)
                    continue
                inference_started = time.monotonic()
                try:
                    response = backend.infer(crop, prompt, pair)
                    talking, confidence = talking_from_response(response)
                except Exception as error:
                    self.get_logger().error(
                        f'VLM inference failed for {pair}: {type(error).__name__}: {error}')
                    response, talking, confidence = str(error), False, 0.0
                inference_elapsed = time.monotonic() - inference_started
                prep, wait, gen, ntok = getattr(
                    backend, 'last_timing', (0.0, 0.0, 0.0, 0))
                self.get_logger().info(
                    f'VLM latency {pair}: {inference_elapsed:.2f}s, '
                    f'crop={crop.shape[1]}x{crop.shape[0]}, tokens={ntok} '
                    f'[tien_xu_ly={prep:.2f}s cho_lock={wait:.2f}s '
                    f'model={gen:.2f}s]')

                # Do not publish a result from an old image. Require a newer
                # camera frame to confirm that both members are still visible
                # in the same scene generation.
                with self.state_lock:
                    validation_sequence = self.latest_observation_sequence
                if not self.pair_visible_after(
                        pair, validation_sequence, item_generation):
                    with self.state_lock:
                        cached = self.interaction_cache.get(pair)
                        if (cached is not None and
                                cached.get('scene_generation') == item_generation):
                            self.interaction_cache.pop(pair, None)
                            self.last_logged_decisions.pop(pair, None)
                    self.finish_vlm_item(pair, item_generation, keep_scene=False)
                    continue

                inference_stamp = self.get_clock().now().to_msg()
                state = ('unknown' if confidence <= 0.0 else
                         'talking' if talking else 'not_talking')
                cached_result = {
                    'state': state,
                    'talking': talking,
                    'confidence': confidence,
                    'response': response,
                    'people': item['people'],
                    'observation_header': item['observation_header'],
                    'inference_stamp': inference_stamp,
                    'time': time.monotonic(),
                    'scene_generation': item_generation,
                }
                with self.state_lock:
                    if (self.scene_generation != item_generation or
                            not all(member_id in self.latest_people
                                    for member_id in pair)):
                        if self.vlm_inflight_pairs.get(pair) == item_generation:
                            self.vlm_inflight_pairs.pop(pair, None)
                        continue
                    settled = self.settled_result(
                        self.interaction_cache.get(pair), cached_result)
                    self.interaction_cache[pair] = settled
                    if self.vlm_inflight_pairs.get(pair) == item_generation:
                        self.vlm_inflight_pairs.pop(pair, None)
                    scene = self.last_vlm_scenes.get(pair)
                    if (scene is not None and
                            scene['scene_generation'] == item_generation):
                        scene['time'] = time.monotonic()
                    # Log while holding the generation lock so a clear event
                    # cannot be printed before this older decision. Report the
                    # settled decision, not the raw answer, so the interaction
                    # topic and terminal log always agree.
                    if bool(self.get_parameter('log_vlm_results').value):
                        self.log_vlm_result(pair, settled)

    def settled_result(self, previous, fresh):
        """Decide what a new answer does to an already-confirmed conversation.

        The model is asked a borderline question and answers in free text, and
        it does not answer in one stable format: a single session here produced
        `{"talking": "có"}`, the unterminated `{"talking": "có}`, and
        `{"talking":true}`. An answer that cannot be read carries no
        information about the conversation, yet it used to overwrite the cache
        exactly like a confident "no". The diagnostic talking state therefore
        still requires `negative_answers_to_clear` contrary answers in a row.
        This cache is legacy-only; it does not control navigation output.
        """
        if previous is None or previous['state'] != 'talking':
            return fresh
        if fresh['state'] == 'talking':
            return fresh

        negatives = previous.get('negatives', 0)
        if fresh['state'] == 'not_talking':
            negatives += 1
            if negatives >= self.negatives_to_clear:
                fresh['negatives'] = negatives
                return fresh
            reason = (f'phủ định {negatives}/{self.negatives_to_clear}, '
                      'chưa đủ để xoá')
        else:
            reason = 'không đọc được câu trả lời'
        self.get_logger().info(
            f'GIỮ KẾT QUẢ VLM ({reason}): {fresh["response"]!r}')
        held = dict(previous)
        held['negatives'] = negatives
        held['time'] = fresh['time']
        held['inference_stamp'] = fresh['inference_stamp']
        return held

    def publish_social_outputs(self):
        now = time.monotonic()
        with self.state_lock:
            people = dict(self.latest_people)
            header = self.latest_header
            # Zero keeps a diagnostic VLM decision until a newer answer arrives.
            # Any positive timeout expires only that interaction record; the
            # person tracking output is unaffected.
            expired = [
                pair for pair, result in self.interaction_cache.items()
                if (self.interaction_timeout > 0.0 and
                    result['state'] != 'processing' and
                    now - result['time'] > self.interaction_timeout)
            ]
            for pair in expired:
                del self.interaction_cache[pair]
            # A cached decision is meaningful only while every member of that
            # pair is present in the latest successful camera observation.
            results = {
                pair: result for pair, result in self.interaction_cache.items()
                if all(member_id in people for member_id in pair)
            }
        if header is None:
            return

        interactions = TalkingInteractions()
        interactions.header = header
        for pair, result in sorted(results.items()):
            interaction = TalkingInteraction()
            interaction.id = f'talking_{pair[0]}_{pair[1]}'
            interaction.observation_header = result['observation_header']
            interaction.inference_stamp = result['inference_stamp']
            interaction.member_ids = list(pair)
            interaction.people = result['people']
            interaction.state = result['state']
            interaction.talking = bool(result['talking'])
            interaction.confidence = float(result['confidence'])
            interaction.center = self.center_of(result['people'])
            interaction.raw_response = result['response']
            interactions.interactions.append(interaction)

        interaction_signature = tuple(
            (item.id, item.state, item.talking, round(item.confidence, 3),
             item.raw_response, item.inference_stamp.sec,
             item.inference_stamp.nanosec)
            for item in interactions.interactions)
        # Publish decisions immediately when they change. An unchanged state
        # gets only a slow heartbeat so `ros2 topic echo` remains readable.
        if (interaction_signature != self.last_interactions_signature or
                now - self.last_interactions_publish >= 10.0):
            self.interactions_pub.publish(interactions)
            self.last_interactions_signature = interaction_signature
            self.last_interactions_publish = now

    def log_vlm_result(self, pair, result):
        if result['state'] == 'talking':
            decision = 'CÓ NÓI CHUYỆN'
        elif result['state'] == 'not_talking':
            decision = 'KHÔNG NÓI CHUYỆN'
        else:
            decision = 'KHÔNG XÁC ĐỊNH'
        # Continuous monitoring may produce the same answer repeatedly. Print
        # only transitions so the decision remains easy to spot in a terminal.
        if self.last_logged_decisions.get(pair) == decision:
            return
        self.last_logged_decisions[pair] = decision
        self.get_logger().info(f'KẾT QUẢ: {decision}')

    @staticmethod
    def angular_distance(first, second):
        return abs(math.atan2(math.sin(first - second),
                              math.cos(first - second)))

    @staticmethod
    def center_of(people):
        center = Point()
        count = float(len(people))
        center.x = sum(person.pose.position.x for person in people) / count
        center.y = sum(person.pose.position.y for person in people) / count
        center.z = sum(person.pose.position.z for person in people) / count
        return center

    def destroy_node(self):
        self.stop_event.set()
        if self.vlm_thread is not None:
            try:
                self.vlm_thread.join(timeout=1.0)
            except KeyboardInterrupt:
                # A second SIGINT from ros2 launch can arrive while the daemon
                # worker is leaving its queue wait. Shutdown must stay clean.
                pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SocialVlmPerception()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
