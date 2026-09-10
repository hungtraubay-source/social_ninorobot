"""The topics-to-observation bridge, shared by training and deployment.

Both the Gym environment and the agent node build their input through this one
module. Duplicating the TF maths in two places is how a policy ends up being fed
people in the map frame during deployment after being trained on people in the
robot frame -- an error nothing crashes on.

Deliberately free of gymnasium, gazebo_msgs and robot_localization: this is the
half of the package the real robot loads, and the robot has none of them
installed.
"""

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from social_perception.msg import (ConstraintField as ConstraintFieldMsg,
                                   People, Zone as ZoneMsg,
                                   ZoneSample as ZoneSampleMsg)
from tf2_ros import TransformException

from social_rl.constraint_field import compile_zones, intrusion_at_zones
from social_rl.observation import (ObservationInput, RelativeEntity,
                                   build_observation)


@dataclass
class EnvConfig:
    """Wiring and episode rules. Distances in metres, times in seconds.

    Shared by both sides: training reads all of it, the agent node reads only
    the topics, the speed limits and control_period. The rest describes how
    episodes are reset in Gazebo, and is carried along so that the file saved
    next to a checkpoint records exactly what produced it.
    """

    # Gazebo's diff-drive plugin listens on /cmd_vel_safe and
    # social_velocity_filter is what normally bridges /cmd_vel to it. Training
    # writes to /cmd_vel_safe directly on purpose: with the filter in the way
    # the executed command is not the sampled action, and the policy would be
    # learning from somebody else's decisions. The deployment node publishes to
    # /cmd_vel instead, so the filter stays as the last line of defence there.
    cmd_vel_topic: str = '/cmd_vel_safe'
    scan_topic: str = '/scan'

    # Where block D's compiled zones go out. Block F is the consumer; RViz and
    # anybody debugging a run are the other two. Published from wherever the
    # zones were compiled, which is once per control step either in training or
    # in the deployment agent.
    constraint_field_topic: str = '/social_rl/constraint_field'

    # Where blocks B and C come from.
    #
    #   ground_truth   Gazebo itself, through social_rl/ground_truth.py. The
    #                  positions, velocities, facings and scene_types are
    #                  exact, so a bad episode is the policy's fault and
    #                  nothing else's. Training only -- it reads /model_states.
    #   perception     the /people topic that social_perception publishes from
    #                  YOLO and depth. This is what the robot has, and what
    #                  the deployment agent always uses.
    #
    # The observation is identical either way, which is the point: a policy
    # trained on ground truth runs unmodified on the real tracker.
    people_source: str = 'ground_truth'
    people_topic: str = '/people'

    # --- ground truth chỉ trong tầm camera ---
    # Ground truth của Gazebo liệt kê MỌI actor, không qua camera, không qua
    # góc nhìn. Train thẳng trên đó thì policy sống trong một thế giới nơi
    # người sau lưng LUÔN được biết, rồi lên robot thật nó lái vào một thế
    # giới mù 274 độ phía sau. Nó chưa bao giờ phải học giữ khoảng cách với
    # chỗ mình không nhìn thấy, vì trong lúc train chuyện đó không tồn tại.
    #
    # true = lọc người theo đúng nón camera trước khi đưa vào observation.
    # Vị trí, vận tốc và scene_type của những người CÒN LẠI vẫn chính xác
    # tuyệt đối - vẫn tách được lỗi policy khỏi lỗi detector, chỉ bỏ đi lợi
    # thế mà robot thật không có.
    people_camera_only: bool = True
    # 1.50098 rad = 86 độ, đúng <horizontal_fov> trong depth_sensor.urdf.xacro.
    camera_fov: float = 1.50098
    # 12.0 m là <clip><far> của cùng cảm biến đó.
    camera_max_range: float = 12.0
    # Camera nằm TRƯỚC trục bánh 0.10 m (depth_sensor_pose trong
    # 2wd_properties.urdf.xacro). Nhỏ, nhưng ở cự ly gần nó dịch biên nón vài
    # độ, và người sát bên hông là đúng chỗ chuyện đó quyết định thấy hay không.
    camera_offset_x: float = 0.10

    # --- che khuất, CHỈ ÁP CHO GROUND TRUTH ---
    # Nón camera bịt 274 độ phía sau; cái này bịt nốt người đứng sau tường hay
    # sau lưng người khác. Hai thứ cộng lại mới ra được tầm nhìn robot thật có.
    # Không áp cho /people: trên robot thật, người bị bàn che nửa dưới vẫn
    # được camera nhìn thấy và YOLO báo đúng - bỏ họ đi là xoá một người có
    # thật. observe() là chỗ chặn chuyện đó.
    people_occlusion: bool = True
    # Bề ngang một người, dùng làm bề rộng góc để xét chắn. Chắn theo CẢ bề
    # ngang chứ không theo một tia, để chân ghế không xoá được cả một người.
    occlusion_person_width: float = 0.5
    # Tia phải dừng sớm hơn người ÍT NHẤT ngần này mới tính là chắn. Bức tường
    # ngay sau lưng người không phải vật cản.
    occlusion_margin: float = 0.3
    # Quá nửa bề ngang bị chắn thì coi như không nhìn thấy.
    occlusion_blocked_fraction: float = 0.5

    # --- BỘ NHỚ NGƯỜI ---
    # Người ra khỏi tầm nhìn vẫn ở lại observation thêm ngần này giây, vị trí
    # được ngoại suy theo vận tốc hằng. Reward ground-truth nay dùng danh sách
    # đầy đủ riêng; bộ nhớ cung cấp cho LSTM bằng chứng về người vừa bị khuất.
    # Trước thay đổi đó, không có memory thì quay lưng từng xoá được tiền phạt:
    # đo giữa cặp talking, 9/13 hướng phạt 0 dù intrusion thật luôn 1.000.
    #
    # 2.0 s khớp chân trời dự báo của khối D. Sai số ngoại suy tăng |v| mỗi
    # giây nên đừng nới rộng; robot thật coast lâu hơn nhiều (tracking_timeout
    # 6.0 s trong social_vlm_perception.yaml) nhưng nó coast bằng quan sát
    # thật, còn đây là suy đoán thuần.
    #
    # 0.0 để tắt hoàn toàn - dùng khi muốn tái hiện lại lỗ hổng để so.
    people_memory_time: float = 2.0
    # Người ĐANG ĐỨNG YÊN lúc nhìn thấy lần cuối được nhớ lâu hơn hẳn, vì lý do
    # duy nhất bắt cửa sổ trên phải ngắn là sai số ngoại suy - mà người đứng
    # yên thì sai số đó bằng 0. Ngưỡng "đứng yên" lấy thẳng từ
    # observation.constraint_field.still_speed, KHÔNG khai lại ở đây.
    #
    # 6.0 s bằng tracking_timeout của social_vlm_perception, tức đúng khoảng
    # robot thật giữ một track đã mất.
    #
    # Đánh đổi đã biết: người đang đứng rồi bỏ đi trong lúc robot không nhìn sẽ
    # để lại một vùng ma ở chỗ cũ tới 6 s. Với `talking`/`static_pair` (người
    # đứng suốt tập) thì không xảy ra; với `crossing` thì người đang đi nên rơi
    # vào cửa sổ 2.0 s ở trên.
    people_memory_time_still: float = 6.0

    # --- KHỐI C GIẢ LẬP ---
    # Khối C chưa tồn tại, nhưng policy sẽ ăn output của nó thì đang được train
    # NGAY BÂY GIỜ trên một ground truth có scene_type luôn đúng và luôn tức
    # thời. Train như thế rồi cắm VLM thật vào là cách để phát hiện, đúng vào
    # lúc tệ nhất, rằng policy đã học cách TIN TUYỆT ĐỐI vào nhãn.
    #
    # NHÃN BỊ LÀM NHIỄU, TIỀN PHẠT THÌ KHÔNG. observe() tính reward trên nhãn
    # thật và danh sách ground-truth đầy đủ. Nón camera và che khuất vẫn giới
    # hạn observation, nhưng không còn làm người biến mất khỏi reward.
    # Phạt theo nhãn sai là dạy policy rằng một cuộc nói chuyện bị đọc nhầm
    # thành `walking` thì thật sự rẻ. Xem ground_truth.SimulatedVLM.
    #
    # MẶC ĐỊNH TẮT. Bật theo curriculum: train sạch cho tới khi policy biết
    # lái, rồi mới bật và --resume. scene_type chỉ chọn hệ số nhân nên shape
    # mạng không đổi, checkpoint nạp lại được.
    vlm_noise: bool = False
    # Nhịp ra phán quyết. Giữ nhãn theo track giữa hai lần, nên một tham số này
    # sinh ra CẢ độ trễ LẪN nhấp nháy. 0.5 s ~ VLM video chạy 2 Hz.
    vlm_period: float = 0.5
    # Đọc nhầm sang một nhãn khác. 0.2 ~ VLM 4-5 lớp đúng 70-85%.
    vlm_wrong_label_prob: float = 0.2
    # Không dám kết luận -> trả '' -> khối D rơi về bộ tỉ lệ trung tính.
    vlm_abstain_prob: float = 0.1
    # Sai số hướng nhìn, radian. Giữ nguyên trong suốt một phán quyết chứ không
    # bốc lại mỗi bước - sai lệch của một model là nhất quán, còn nhiễu mỗi
    # bước thì trung bình lại thành đúng.
    vlm_facing_sigma: float = 0.0
    # Mất hẳn hướng nhìn. Tái hiện ĐÚNG khối B hiện nay: orientation 0 trong
    # frame `map` cho người đứng yên - tức một hướng la bàn cố định, KHÔNG
    # phải vùng đối xứng. Vùng xoay sai còn tệ hơn vùng không có.
    vlm_facing_lost_prob: float = 0.0
    # 0 = ngẫu nhiên mỗi lần chạy. Đặt khác 0 để lặp lại đúng một run.
    vlm_seed: int = 0

    # Kept so old saved configs still load. Ground-truth training now always
    # computes the complete-list intrusion because reward depends on it; false
    # can no longer disable that invariant. The policy never receives this
    # list, while TensorBoard still records it for the visible-vs-full check.
    measure_hidden_intrusion: bool = True
    ground_truth_people_topic: str = '/social_gt/people'
    model_states_topic: str = '/model_states'
    scenario_topic: str = '/animated_people/scenario'
    # The EKF output (gazebo.launch.py odom_topic, and the same name on the
    # robot). Read for the measured v and w that go into the observation: the
    # last command says what was asked for, this says what the base is doing.
    odom_topic: str = '/odom'
    robot_frame: str = 'base_link'
    # Frame the goals and start poses are written in, and the frame the goal is
    # carried out of. `odom` (default) keeps RL independent of Nav2: that edge
    # comes from the EKF, which is running anyway. Use `map` only when AMCL is
    # up -- with no localizer, map -> base_link does not exist and every step
    # fails on the lookup.
    #
    # The two share an origin by construction: the EKF starts at the spawn
    # pose, and gazebo.launch.py publishes world -> map from that same spawn
    # pose. So a goal keeps its numbers when you switch this over.
    goal_frame: str = 'odom'

    # 5 Hz. The lidar runs at 10 Hz and the depth camera at 8 Hz, so a faster
    # control rate would mostly re-observe the same frame twice.
    control_period: float = 0.2
    # A scan received once at startup is not valid forever. This watchdog is
    # used by deployment and training before any policy step is allowed.
    scan_timeout: float = 0.5
    max_episode_steps: int = 250
    # A /people message older than this counts as "nobody in sight" rather than
    # as a person frozen in place. Perception drops frames while the VLM works.
    people_timeout: float = 1.0
    # Odometry arrives at 30 Hz or better on both sides. Anything older than
    # this means the EKF stopped, and a stale velocity would tell the policy
    # the robot is still moving after it has stopped.
    odom_timeout: float = 0.5
    # How long reset() waits for the first scan and a usable TF tree.
    startup_timeout: float = 60.0

    max_linear_speed: float = 0.5
    max_angular_speed: float = 1.0
    # Reverse is disabled: the base has no rear sensor, so backing up is blind.
    # The policy turns and drives instead, which is also what the real one must
    # do.
    allow_reverse: bool = False

    # --- simulation-only: episode reset ---
    randomize_start: bool = True
    entity_name: str = 'linorobot2'
    set_entity_state_service: str = '/set_entity_state'
    ekf_set_pose_service: str = '/set_pose'
    reseed_localization: bool = True
    # Map-frame [x, y, yaw]. Defaults sit in the open part of cafe_vlm.pgm on
    # the robot's side of the two actors.
    start_poses: list = field(default_factory=lambda: [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.6], [1.0, 0.0, 0.3], [0.0, 1.5, -0.4]])
    # Map-frame [x, y] goals. Reaching one ends the episode. The defaults lie
    # beyond the pair of actors at map (3.0, 2.8) and (3.0, 1.2), so the direct
    # line to them runs through the conversation and going around costs
    # distance -- which is the whole trade-off being learned.
    goals: list = field(default_factory=lambda: [
        [4.5, 2.0], [4.0, 3.5], [4.5, 0.8], [3.8, 1.5]])
    # A goal closer than this makes for an episode with no avoidance in it.
    minimum_goal_distance: float = 2.0
    # Upper bound, and only meaningful when free_space_map draws the pair. It
    # is a budget, not a horizon: an episode is max_episode_steps long and the
    # base covers about a metre every 12 steps, so this is how far the robot
    # can be asked to drive before the episode is over. Whether a goal that far
    # is reachable at all is a property of the MAP -- measured on cafe_vlm, the
    # walked path is only 1.06x the straight line, so nothing there needs a
    # global plan. A map with real corridors would.
    maximum_goal_distance: float = 8.0
    # '<package>/<path to a map_server .yaml>'. Set, both ends of the episode
    # are drawn from the free space of that map instead of from the two lists
    # above -- which are 16 fixed routes, all of them 3.18-5.80 m long and
    # bearing -30 to +57 degrees, while a goal picked in RViz is neither.
    # Empty keeps the lists, which is what every run before 01-09-2026 used.
    free_space_map: str = ''

    # One of these is drawn at the start of every episode and sent to the
    # actor plugin, which lays the people out afresh around the robot's route.
    # Repeat a name to weight it. `none` is an episode with nobody in it, kept
    # in the mix so the policy does not learn that there is always somebody to
    # avoid and stop trusting an empty field.
    # Three situations since 10-09-2026: `talking`, `passing` (crossing and
    # approaching merged), and `none`. rl_train.yaml sets the real mix.
    scenarios: list = field(default_factory=lambda: [
        'none', 'none', 'none', 'talking', 'passing', 'passing'])
    # Seconds of simulated time to let the actors settle into the new scene
    # before the first observation. Walkers start at the edge and the plugin
    # needs a publish period or two to report a velocity for them at all.
    scenario_settle_time: float = 0.6

    # Hold physics between control steps: unpause, let exactly control_period
    # of simulated time pass, pause again, and only then read the observation.
    #
    # Two things this buys, both measured rather than assumed:
    #
    #   * one step means one control period. Unpaused, a PPO update stops this
    #     node from spinning for seconds of wall time while Gazebo keeps
    #     running, and the first step afterwards was measured reading /odom
    #     1.302 s old. With physics held, the world simply does not move while
    #     the network is being updated.
    #   * training stops depending on how fast this machine happens to be, so
    #     a run is comparable with the one before it.
    #
    # The cost is two service round trips per step; the overshoot they cause is
    # logged per episode as `overshoot`, so it is a number you can look at
    # rather than a worry.
    pause_between_steps: bool = True

    # Open the Gazebo 3D window and watch the robot drive while it learns.
    # Costs real time: gzclient is the largest CPU consumer in the simulation,
    # and every second it takes is a second of training not happening. Leave it
    # off for long runs, turn it on to see what the policy is actually doing.
    render: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> 'EnvConfig':
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f'unknown env keys: {sorted(unknown)}. '
                f'Valid keys: {sorted(known)}')
        return cls(**values)


def yaw_to_quaternion(yaw: float):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def visible_to_camera(x: float, y: float, env_config) -> bool:
    """Whether the depth camera could see somebody at robot-frame (x, y).

    One function for both people sources on purpose. It models the SENSOR, not
    where the numbers came from: block B on the robot cannot see behind itself,
    so training on a ground truth that can hands the policy a sense it will not
    have. Applying the same cone to the perception path as well keeps the two
    honest about each other -- and it is not a no-op there either, because
    social_vlm_perception coasts a lost track for tracking_timeout seconds and
    would otherwise keep reporting somebody who walked out of frame.

    Field of view and range only. Whether the line of sight is clear is the
    separate question occluded_by_scan() answers.
    """
    offset_x = x - env_config.camera_offset_x
    if math.hypot(offset_x, y) > env_config.camera_max_range:
        return False
    return abs(math.atan2(y, offset_x)) <= 0.5 * env_config.camera_fov


def occluded_by_scan(x, y, scan_xy, env_config) -> bool:
    """Whether the laser says something solid stands in front of (x, y).

    The other half of the gap between Gazebo's ground truth and a camera: the
    cone filter takes away the 274 degrees behind the robot, this takes away
    the person standing behind a wall or behind somebody else. Without it the
    policy can still be handed a conversation it has no way of seeing, and it
    learns to swerve around walls.

    GROUND TRUTH ONLY. It must never be applied to the /people topic, and
    observe() is where that is enforced. On the real robot a person detected
    over a table is a correct detection -- the camera sees their upper body
    while the lidar plane sees the table -- and dropping them because the beam
    stopped short would delete a person who is really there.

    Blockage is judged over the whole width of a person rather than on one
    beam, so a chair leg does not erase somebody standing behind it: the
    fraction of beams across their shoulders that stop at least
    occlusion_margin short has to exceed occlusion_blocked_fraction.
    """
    distance = math.hypot(x, y)
    if distance <= env_config.occlusion_margin or scan_xy.shape[0] == 0:
        return False
    bearing = math.atan2(y, x)
    # Half the angle a person subtends at that range. Capped so that somebody
    # almost touching the robot does not claim the entire scan.
    half_width = min(math.atan2(0.5 * env_config.occlusion_person_width,
                                distance), 0.5 * math.pi)
    beams = np.arctan2(scan_xy[:, 1], scan_xy[:, 0]) - bearing
    beams = np.abs(np.arctan2(np.sin(beams), np.cos(beams)))
    across = beams <= half_width
    if not across.any():
        # Nothing measured in that direction at all: open space out to
        # scan_max_range, or beyond the lidar's reach. Either way not blocked.
        return False
    blocking = np.hypot(scan_xy[across, 0], scan_xy[across, 1]) < (
        distance - env_config.occlusion_margin)
    return bool(blocking.mean() > env_config.occlusion_blocked_fraction)


def occluded_by_person(person, others, env_config) -> bool:
    """Whether a nearer person stands on the line of sight to this one.

    The scan cannot answer this one. Gazebo actors are not in the ray-cast
    space -- /scan passes straight through them, which is documented at length
    in RUN_RL.txt -- so somebody standing directly behind somebody else is
    invisible to occluded_by_scan() and would otherwise stay in the ground
    truth forever. On the real robot the camera has exactly this blind spot,
    which is the gap being closed.

    Blocked means the blocker's centre falls within half a person-width of the
    sight line, so more than half of the person behind is hidden -- the same
    threshold occluded_by_scan() applies to its beams.
    """
    distance = math.hypot(person.x, person.y)
    if distance <= env_config.occlusion_margin:
        return False
    unit_x, unit_y = person.x / distance, person.y / distance
    half_width = 0.5 * env_config.occlusion_person_width
    for other in others:
        # By track_id, not by identity: `others` is the unfiltered list and
        # `person` was taken from the filtered one, so they are equal entries
        # held in different objects.
        if other.track_id == person.track_id:
            continue
        along = other.x * unit_x + other.y * unit_y
        # Only somebody genuinely in front counts. The margin also keeps the
        # two members of a conversation pair from hiding each other when they
        # happen to be the same distance away.
        if along <= 0.0 or along >= distance - env_config.occlusion_margin:
            continue
        across = abs(-other.x * unit_y + other.y * unit_x)
        if across < half_width:
            return True
    return False


def inverse_transform_point(transform, x: float, y: float):
    """Undo transform_point: robot frame back to the frame it came from."""
    yaw = quaternion_to_yaw(transform.rotation)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    dx = x - transform.translation.x
    dy = y - transform.translation.y
    return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy)


class PeopleMemory:
    """Keeps somebody in the observation briefly after they leave the view.

    Why this was added: the camera cone and the occlusion test are both applied
    before the visible constraint field is compiled. Before full-scene reward,
    a person who slid out of the 86 degree cone took their region -- and the
    penalty for standing in it -- with them.
    Measured 29-08-2026 by rotating a stationary robot in place: parked between
    two people 0.75 m away and talking to each other, the true intrusion is
    1.000 at every heading, but 9 of 13 headings charged exactly 0.00, because
    pointing the camera along the axis of the conversation puts both people
    outside the cone. Turning away was a way to make the fee disappear without
    moving.

    So memory is not "let the policy see through walls". It is: a robot that
    just watched two people start a conversation does not forget them the
    instant it turns its head. social_vlm_perception already coasts a lost
    track for tracking_timeout (6.0 s) for exactly this reason, so the real
    robot has this behaviour whether or not training does.

    Applied to BOTH observation sources. Ground-truth reward now bypasses this
    filtered/remembered list and charges the complete current scene; memory is
    what gives the LSTM usable evidence about recently hidden people.

    Positions are held in the goal frame, not the robot frame. The robot frame
    moves with the robot, so a remembered person stored in it would slide
    across the room every time the base did.
    """

    def __init__(self, env_config, observation_config):
        self._env = env_config
        # Only for still_speed, which decides which memory window applies.
        # Read from block D's config so there is one definition of "standing
        # still" in the package rather than two that can drift apart.
        self._observation = observation_config
        # track_id -> (stamp, x, y, vx, vy, facing, scene_type, confidence),
        # everything but the stamp in the goal frame.
        self._seen = {}

    def clear(self):
        """Forget everybody. Called on reset: people from the last episode are
        not evidence about this one."""
        self._seen.clear()

    def remember(self, visible, transform, now: float):
        """Merge the currently visible people with recently seen ones.

        `transform` is goal_frame -> robot_frame, the same one the goal is
        carried across. `now` is seconds on the node clock, which is simulated
        time during training.
        """
        window = max(self._env.people_memory_time,
                     self._env.people_memory_time_still)
        if window <= 0.0:
            return visible

        for person in visible:
            fixed_x, fixed_y = inverse_transform_point(
                transform, person.x, person.y)
            fixed_vx, fixed_vy = rotate_vector(
                transform, person.vx, person.vy, inverse=True)
            self._seen[person.track_id] = (
                now, fixed_x, fixed_y, fixed_vx, fixed_vy,
                person.facing - quaternion_to_yaw(transform.rotation),
                person.scene_type, person.scene_confidence)

        present = {person.track_id for person in visible}
        transform_yaw = quaternion_to_yaw(transform.rotation)
        recalled = []
        for track_id, entry in list(self._seen.items()):
            stamp, fixed_x, fixed_y, fixed_vx, fixed_vy = entry[:5]
            elapsed = now - stamp
            # Somebody who was standing still when last seen is remembered for
            # longer: constant-velocity coasting has nothing to get wrong about
            # them, and the o-space of a conversation is exactly the region the
            # robot must not be allowed to wait out.
            still = (math.hypot(fixed_vx, fixed_vy)
                     <= self._observation.constraint_field.still_speed)
            limit = (self._env.people_memory_time_still if still
                     else self._env.people_memory_time)
            if elapsed > limit or elapsed < 0.0:
                # elapsed < 0 is a clock that went backwards, which on this
                # stack means the episode reset while an entry was still held.
                del self._seen[track_id]
                continue
            if track_id in present:
                continue
            # Constant velocity, the same assumption block D already makes for
            # its prediction channels. The error grows at |v| per second, which
            # is why people_memory_time is short rather than generous.
            x, y = transform_point(transform,
                                   fixed_x + fixed_vx * elapsed,
                                   fixed_y + fixed_vy * elapsed)
            vx, vy = rotate_vector(transform, fixed_vx, fixed_vy)
            recalled.append(RelativeEntity(
                x=x, y=y, vx=vx, vy=vy,
                facing=entry[5] + transform_yaw,
                scene_type=entry[6], track_id=track_id,
                scene_confidence=entry[7]))
        return visible + recalled


def scale_action(action, env_config):
    """Map a policy action in [-1, 1]^2 onto (linear, angular) in SI units.

    Shared with the agent node for the same reason the observation is: the
    weights were fitted against this exact mapping, and a deployment that
    scales differently drives a policy that was never trained.
    """
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1),
                     -1.0, 1.0)
    if env_config.allow_reverse:
        linear = float(action[0]) * env_config.max_linear_speed
    else:
        # Forward-only: [-1, 1] is stretched onto [0, max_linear_speed].
        linear = (float(action[0]) + 1.0) / 2.0 * env_config.max_linear_speed
    return linear, float(action[1]) * env_config.max_angular_speed


def transform_point(transform, x: float, y: float):
    """Apply the 2-D part of a TransformStamped.transform to a point."""
    yaw = quaternion_to_yaw(transform.rotation)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return (transform.translation.x + cos_yaw * x - sin_yaw * y,
            transform.translation.y + sin_yaw * x + cos_yaw * y)


def rotate_vector(transform, x: float, y: float, inverse: bool = False):
    """Rotate a velocity into the target frame (translation does not apply).

    inverse=True rotates the other way, which is what PeopleMemory needs to
    put a robot-frame velocity back into the frame it stores people in.
    """
    yaw = quaternion_to_yaw(transform.rotation)
    if inverse:
        yaw = -yaw
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return (cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y)


class PerceptionBridge:
    """Latest /scan, /people and /odom, expressed in the robot frame."""

    def __init__(self, node, tf_buffer, env_config, observation_config,
                 people_provider=None):
        self._node = node
        self._tf_buffer = tf_buffer
        self._env = env_config
        self._observation = observation_config
        self._logger = node.get_logger()
        # Where the people in the observation come from. None means the /people
        # topic below, which is the only source the robot has; training passes
        # a GroundTruthPeople instead. Everything downstream of this line is
        # identical either way -- that is what makes one policy run on both.
        self._people_provider = people_provider
        # Shared by both sources on purpose: the observation has to be the same
        # whether a person came from Gazebo or from YOLO, and that includes how
        # long they survive after leaving the frame.
        self._memory = PeopleMemory(env_config, observation_config)

        # Keep only the newest sample: an RL step acts on now, and a queue of
        # stale scans would just delay every reaction by its own length.
        sensor_qos = QoSProfile(
            depth=1, history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE)
        self.scan = None
        self.people = None
        self.odom = None
        self.people_transform_valid = True
        node.create_subscription(LaserScan, env_config.scan_topic,
                                 self._on_scan, sensor_qos)
        node.create_subscription(People, env_config.people_topic,
                                 self._on_people, 10)
        node.create_subscription(Odometry, env_config.odom_topic,
                                 self._on_odom, sensor_qos)

        # Block D's output, published from the one place it is already
        # compiled. Block F subscribes to this rather than compiling its own
        # copy: two compilations of the same people list drift the moment one
        # of them is handed a message the other did not see, and a shield
        # arguing with the policy about where a region is defeats the point of
        # having a shield.
        self._field_pub = node.create_publisher(
            ConstraintFieldMsg, env_config.constraint_field_topic, 10)

    def _on_scan(self, msg):
        self.scan = msg

    def _on_people(self, msg):
        self.people = msg

    def _on_odom(self, msg):
        self.odom = msg

    # A stamp slightly ahead of this node's view of the clock is not an error:
    # Gazebo stamps a sensor at the moment it fires, which can be a fraction of
    # a /clock tick later than the last tick this node processed. Measured on
    # this machine: /scan arrives with an age of -0.091 s every single frame.
    # Rejecting that as "not fresh" would stall every wait in the package, so
    # the window is symmetric around zero rather than one-sided.
    def has_scan(self) -> bool:
        if self.scan is None:
            return False
        return abs(self._age(self.scan)) <= self._env.scan_timeout

    def has_odom(self) -> bool:
        if self.odom is None:
            return False
        return abs(self._age(self.odom)) <= self._env.odom_timeout

    def can_transform_map(self) -> bool:
        return self._tf_buffer.can_transform(
            self._env.robot_frame, self._env.goal_frame, Time())

    def _age(self, msg) -> float:
        stamp = Time.from_msg(msg.header.stamp)
        return (self._node.get_clock().now() - stamp).nanoseconds * 1e-9

    def _lookup(self, source_frame):
        return self._tf_buffer.lookup_transform(
            self._env.robot_frame, source_frame, Time()).transform

    def scan_returns(self):
        """Usable laser returns as robot-frame points and their ranges.

        The beams are carried across TF rather than assumed to start at
        base_link: the lidar sits forward of the wheel axis on this base, and
        rasterising from the wrong origin shifts every obstacle by that offset
        -- a bias the policy would learn to compensate for in simulation and
        then get wrong on any robot mounted differently.

        One pass, three consumers -- the occupancy channel, the collision check
        and the occlusion test -- so they cannot disagree about which beams are
        real. They did: minimum_scan() used to read raw ranges while the grid
        read transformed points, and both were counting the robot's own base.

        Returns (points (N, 2), ranges (N,)), both float32 and aligned.
        """
        scan = self.scan
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        angles = (scan.angle_min
                  + np.arange(ranges.size, dtype=np.float32)
                  * scan.angle_increment)
        limit = min(float(scan.range_max), self._observation.scan_max_range)
        usable = (np.isfinite(ranges) & (ranges >= float(scan.range_min))
                  & (ranges <= limit))
        ranges = ranges[usable]
        angles = angles[usable]
        local_x = ranges * np.cos(angles)
        local_y = ranges * np.sin(angles)

        transform = self._lookup(scan.header.frame_id)
        yaw = quaternion_to_yaw(transform.rotation)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        points = np.stack(
            [transform.translation.x + cos_yaw * local_x - sin_yaw * local_y,
             transform.translation.y + sin_yaw * local_x + cos_yaw * local_y],
            axis=1).astype(np.float32)

        # A return inside the robot's own footprint is the robot. Tested on the
        # transformed point rather than on the raw range, because the lidar is
        # not at base_link and a plain range floor would cut a different disc
        # in front than behind. Nothing real can be there, so this cannot hide
        # an obstacle -- see ObservationConfig.scan_self_hit_radius.
        outside = (np.hypot(points[:, 0], points[:, 1])
                   >= self._observation.scan_self_hit_radius)
        return points[outside], ranges[outside]

    def scan_points(self):
        """Laser returns as (N, 2) points in the robot frame."""
        return self.scan_returns()[0]

    def minimum_scan(self, ranges=None) -> float:
        """Closest laser return, clipped. What the collision check reads.

        Pass the ranges from scan_returns() when they are already to hand; the
        answer has to come off the same filtered set the grid was drawn from.
        """
        if ranges is None:
            ranges = self.scan_returns()[1]
        limit = self._observation.scan_max_range
        if ranges.size == 0:
            return limit
        return float(min(float(ranges.min()), limit))

    def robot_velocity(self):
        """Measured (linear, angular) from the EKF, in the robot frame."""
        if not self.has_odom():
            age = 'never received' if self.odom is None else f'{self._age(self.odom):+.3f} s old'
            raise RuntimeError(
                f'no fresh message on {self._env.odom_topic} ({age}, limit '
                f'{self._env.odom_timeout:.2f} s). The EKF publishes it; '
                f'without odometry the observation has no robot state.')
        twist = self.odom.twist.twist
        return float(twist.linear.x), float(twist.angular.z)

    def relative_people(self):
        self.people_transform_valid = True
        message = self.people
        if message is None or not message.people:
            return []
        # An old message is treated as nobody in sight, not as people frozen
        # where they were: perception stops publishing while it drops frames,
        # and a stale position is worse than none.
        if self._age(message) > self._env.people_timeout:
            return []
        try:
            transform = self._lookup(message.header.frame_id)
        except TransformException as error:
            self.people_transform_valid = False
            self._logger.warn(f'people TF lookup failed: {error}',
                              throttle_duration_sec=5.0)
            return []
        people = []
        transform_yaw = quaternion_to_yaw(transform.rotation)
        for person in message.people:
            x, y = transform_point(transform, person.pose.position.x,
                                   person.pose.position.y)
            vx, vy = rotate_vector(transform, person.velocity.linear.x,
                                   person.velocity.linear.y)
            # Facing and scene_type are what block C fills in. A tracker that
            # publishes neither leaves facing at 0 and scene_type empty, and
            # block D falls back to its neutral region -- the observation keeps
            # its shape either way, so a policy does not stop loading the day
            # the VLM is switched off.
            if (self._env.people_camera_only
                    and not visible_to_camera(x, y, self._env)):
                continue
            people.append(RelativeEntity(
                x, y, vx, vy,
                facing=quaternion_to_yaw(person.pose.orientation) + transform_yaw,
                scene_type=person.scene_type,
                track_id=person.id,
                scene_confidence=person.scene_confidence))
        return people

    def _publish_field(self, field):
        """Put block D's zones on the wire for block F and for RViz.

        Frame is the robot frame, which is what the zones are already in, so a
        consumer needs no TF to act on them. Stamped with the node clock rather
        than with a sensor stamp: the zones are a statement about now, compiled
        from whatever the newest people message was, and dating them by that
        message would make a shield reject its own input as stale during the
        gap between camera frames.
        """
        message = ConstraintFieldMsg()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.header.frame_id = self._observation.robot_frame
        message.horizon_steps = int(field.horizon_steps)
        message.dt = float(field.dt)
        for zone in field.zones:
            entry = ZoneMsg()
            entry.zone_id = zone.zone_id
            entry.track_ids = list(zone.track_ids)
            entry.scene_type = zone.scene_type
            entry.confidence = float(zone.confidence)
            entry.hardness = zone.hardness
            entry.valid_from = float(zone.valid_from)
            entry.valid_to = float(zone.valid_to)
            for sample in zone.trajectory_of_zone:
                item = ZoneSampleMsg()
                item.t = float(sample.t)
                item.shape = sample.shape
                item.center = [float(value) for value in sample.center]
                item.size = [float(value) for value in sample.size]
                item.core = float(sample.core)
                item.orientation = float(sample.orientation)
                entry.trajectory_of_zone.append(item)
            message.zones.append(entry)
        self._field_pub.publish(message)

    def goal_transform(self):
        """goal_frame -> robot_frame. The goal and PeopleMemory both ride it."""
        try:
            return self._lookup(self._env.goal_frame)
        except TransformException as error:
            raise RuntimeError(
                f'{self._env.goal_frame} -> {self._env.robot_frame} lookup '
                f'failed: {error}. AMCL publishes map -> odom; without '
                f'localization running there is no goal to steer to.')

    def relative_goal(self, goal_x: float, goal_y: float):
        """Carry a goal given in the map frame into the robot frame."""
        return transform_point(self.goal_transform(), goal_x, goal_y)

    def forget_people(self):
        """Drop the memory of everybody. Called between episodes."""
        self._memory.clear()

    def observe(self, goal_map_x: float, goal_map_y: float):
        """Build the policy input and the raw quantities the reward needs."""
        if self.scan is None:
            raise RuntimeError(f'no message on {self._env.scan_topic} yet')
        # Looked up once and reused: the goal and the remembered people are
        # carried across the same edge, and two lookups a step could land on
        # two different transforms.
        goal_transform = self.goal_transform()
        goal_x, goal_y = transform_point(goal_transform, goal_map_x, goal_map_y)
        # Once, up front: the grid, the collision check and the occlusion test
        # all read this one set of beams, with the robot's own base already
        # taken out of it.
        scan_xy, scan_ranges = self.scan_returns()
        # None means this bridge has no simulator truth (the perception path);
        # an empty list means ground truth is available and the scene really
        # contains nobody.  The distinction matters now that the full list is
        # an input to the training reward rather than only a diagnostic.
        full_people = None
        if self._people_provider is not None:
            people = self._people_provider.relative_people()
            # The whole scene, cone ignored. Occlusion needs it because
            # somebody already dropped for being out of frame still hides
            # whoever stands behind them.  The training reward also needs this
            # exact unfiltered truth so turning away cannot erase the charge.
            # Fetched once and kept even when it is an empty list.
            full_people = self._people_provider.relative_people(
                apply_camera=False)
            if self._env.people_occlusion:
                # GROUND TRUTH ONLY, and this branch is what keeps it that way:
                # _people_provider is None on the robot. See occluded_by_scan()
                # for why the perception path must not get the same treatment.
                #
                # Walls and furniture come from the laser; people come from
                # each other, because /scan goes straight through a Gazebo
                # actor and cannot see them blocking anything.
                people = [person for person in people
                          if not occluded_by_scan(person.x, person.y, scan_xy,
                                                  self._env)
                          and not occluded_by_person(person, full_people,
                                                     self._env)]
            self.people_transform_valid = True
        else:
            people = self.relative_people()
        # Last, so it sees the list after every visibility rule has run: what
        # is remembered is exactly what was once shown.
        now = self._node.get_clock().now().nanoseconds * 1e-9
        people = self._memory.remember(people, goal_transform, now)
        # Simulated block C, training only. `people` stays the truth; `shown`
        # is what the policy is given. They are the same object unless
        # env.vlm_noise is on.
        shown = people
        corrupted = 0
        if self._people_provider is not None and self._env.vlm_noise:
            shown = self._people_provider.corrupt(people, now)
            corrupted = self._people_provider.corrupted_people
        linear, angular = self.robot_velocity()
        # Block D. The channels come off `shown`, so the policy sees exactly
        # the situation its perception describes -- including a misread one.
        field = compile_zones(shown, self._observation.constraint_field)
        self._publish_field(field)
        observation = build_observation(
            ObservationInput(scan_points=scan_xy,
                             goal_x=goal_x, goal_y=goal_y,
                             people=shown,
                             linear=linear, angular=angular),
            self._observation, field)
        # Keep a filtered field with the TRUE activity labels.  It remains the
        # visible/remembered diagnostic and is the explicit reward fallback
        # when no simulator-only full list exists (people_source=perception).
        #
        # With simulated block C noise, charging by its misread label would
        # teach the policy that a misclassified situation is truly cheaper, so
        # this copy deliberately keeps the uncorrupted labels.
        visible_truth_field = field if shown is people else compile_zones(
            people, self._observation.constraint_field)
        state = {
            'goal_distance': math.hypot(goal_x, goal_y),
            # Angle from the robot's heading to the goal, base_link frame.
            # The heading shaping term in reward.py rewards shrinking |this|,
            # so rotating toward a goal behind the robot earns something
            # before any distance is closed.
            'goal_bearing': math.atan2(goal_y, goal_x),
            'minimum_scan': self.minimum_scan(scan_ranges),
            # Visible/remembered K_soc with TRUE labels, at t = 0.0. This stays
            # separate from the complete-list value below so eval can compare
            # keeping distance with merely keeping people out of the view.
            'social_intrusion': intrusion_at_zones(visible_truth_field),
            # What the policy believed it was in. Equal to the line above
            # unless simulated block C changed a label. Diagnostic only.
            'social_intrusion_shown': intrusion_at_zones(field),
            'corrupted_labels': corrupted,
            # Kept for the episode summary only. It is the number a human reads
            # to judge a rollout ("closest person 1.15 m"), not one the reward
            # uses any more.
            'person_distances': [person.distance for person in people],
            # A non-empty tracker message that cannot be transformed is not
            # equivalent to "nobody there" during deployment. The agent uses
            # this bit to fail closed; the training environment can still
            # expose it for diagnostics without changing its reward contract.
            'people_transform_valid': self.people_transform_valid,
        }
        # Full simulator truth, cone and occlusion ignored.  SocialAvoidEnv
        # charges this value at the same social_penalty as the filtered one;
        # keeping both values also preserves the visible-vs-full peak metric.
        # Test against None rather than truthiness so an empty full scene emits
        # an authoritative 0.0 instead of looking like unavailable truth.
        if full_people is not None:
            hidden_field = compile_zones(
                full_people, self._observation.constraint_field)
            state['social_intrusion_hidden'] = intrusion_at_zones(hidden_field)
            # Clamped: memory can recall somebody the scene no longer lists
            # at all, which would otherwise read as a negative count.
            state['hidden_people'] = max(len(full_people) - len(people), 0)
            # person_distances above is measured on the filtered list, so the
            # episode summary's "closest person" cannot see the person the
            # robot is about to walk into. Measured: an episode reported
            # 2.35 m while the unseen intrusion sat at 1.00.
            state['person_distances_all'] = [
                person.distance for person in full_people]
        return observation, state
