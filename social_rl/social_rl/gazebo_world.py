"""Episode reset against a running Gazebo Classic world.

Only the training environment uses this; the deployment agent never imports it.
That split is deliberate -- teleporting the base is the one thing in the whole
package that cannot exist on hardware, so it lives in a file the robot never
loads.

Resetting the robot is not one call. The diff-drive plugin publishes no TF, EKF
owns odom -> base_footprint and AMCL owns map -> odom, so a base that jumps in
Gazebo without both of them being told about it leaves the whole TF tree
pointing at where the robot used to be: /people lands in the wrong place and
every observation after the reset is quietly wrong.
"""

import math
import time

import rclpy
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.time import Time
from robot_localization.srv import SetPose
from std_srvs.srv import Empty
from tf2_ros import TransformException

from social_rl.ros_interface import quaternion_to_yaw, yaw_to_quaternion


class GazeboWorld:
    """Teleport the base and re-seed the two localizers that follow it."""

    def __init__(self, node, tf_buffer, *, entity_name='linorobot2',
                 set_entity_state_service='/set_entity_state',
                 ekf_set_pose_service='/set_pose',
                 reseed_localization=True, service_timeout=10.0):
        self._node = node
        self._tf_buffer = tf_buffer
        self._entity_name = entity_name
        self._reseed = reseed_localization
        self._timeout = service_timeout

        self._set_state = node.create_client(SetEntityState,
                                             set_entity_state_service)
        self._pause = node.create_client(Empty, '/pause_physics')
        self._unpause = node.create_client(Empty, '/unpause_physics')
        self._set_pose = node.create_client(SetPose, ekf_set_pose_service)
        self._initial_pose_pub = node.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

    def pause(self):
        """Freeze the simulation. Physics, sensors and /clock all stop."""
        self._call(self._pause, Empty.Request())

    def unpause(self):
        self._call(self._unpause, Empty.Request())

    def wait_for_services(self):
        """Block until Gazebo is up. Called once, before the first episode."""
        for client in (self._set_state, self._pause, self._unpause):
            if not client.wait_for_service(timeout_sec=self._timeout):
                raise RuntimeError(
                    f'service {client.srv_name} never appeared. Is gzserver '
                    f'running with libgazebo_ros_init.so and does the world '
                    f'load libgazebo_ros_state.so?')
        if self._reseed and not self._set_pose.wait_for_service(
                timeout_sec=self._timeout):
            raise RuntimeError(
                f'service {self._set_pose.srv_name} never appeared. That is '
                f'the EKF in linorobot2_base; run gazebo.launch.py without '
                f'run_ekf:=false, or set reseed_localization: false.')
        self._wait_for_world_transform()

    def _wait_for_world_transform(self):
        """Spin until the static world -> map edge is in the TF buffer.

        wait_for_service does not spin the node, so at this point nothing has
        processed a single message and the buffer is empty -- the first
        teleport would fail on a transform that is in fact being published.
        /tf_static is latched, so one spin that receives it is enough, but that
        spin has to happen before anybody looks the edge up.
        """
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._tf_buffer.can_transform('world', 'map', Time()):
                return
        raise RuntimeError(
            f'no world -> map transform after {self._timeout:.0f} s. It comes '
            f'from the world_to_map static publisher in gazebo.launch.py, so '
            f'start Gazebo first -- or set randomize_start: false to train '
            f'without moving the robot between episodes.')

    def _call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future,
                                         timeout_sec=self._timeout)
        if not future.done():
            raise RuntimeError(f'{client.srv_name} timed out after '
                               f'{self._timeout} s')
        return future.result()

    def map_to_world(self, x: float, y: float, yaw: float):
        """Convert a map-frame pose into the world frame Gazebo expects.

        gazebo.launch.py publishes world -> map from spawn_x/spawn_y/spawn_yaw,
        because the saved map's origin is wherever the robot started. Reading
        that edge from TF rather than copying the offset into this package's
        config means changing the spawn arguments cannot silently teleport the
        robot to the wrong place.
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                'world', 'map', Time()).transform
        except TransformException as error:
            raise RuntimeError(
                f'no world -> map transform ({error}). It comes from the '
                f'world_to_map static publisher in gazebo.launch.py.')
        offset_yaw = quaternion_to_yaw(transform.rotation)
        cos_yaw, sin_yaw = math.cos(offset_yaw), math.sin(offset_yaw)
        return (transform.translation.x + cos_yaw * x - sin_yaw * y,
                transform.translation.y + sin_yaw * x + cos_yaw * y,
                offset_yaw + yaw)

    def teleport_robot(self, x: float, y: float, yaw: float, z: float = 0.35):
        """Move the base to a pose and tell the localizers about it.

        The pose is read in the frame the goals use (`odom` by default, `map`
        with Nav2 running). The two share an origin -- the EKF starts at the
        spawn pose and world -> map is published from that same spawn pose --
        so the numbers do not change when you switch goal_frame over.

        Without AMCL the /initialpose publish below simply has no subscriber;
        the EKF seed is what matters, and it is what makes odom -> base_link
        agree with the body again.
        """
        world_x, world_y, world_yaw = self.map_to_world(x, y, yaw)

        # Physics is paused around the jump so the wheels are not integrating a
        # command while the body is being moved: unpaused, the base sometimes
        # arrives already sliding and the first observation of the episode is
        # taken mid-skid.
        self._call(self._pause, Empty.Request())
        try:
            request = SetEntityState.Request()
            request.state.name = self._entity_name
            request.state.reference_frame = 'world'
            request.state.pose.position.x = world_x
            request.state.pose.position.y = world_y
            request.state.pose.position.z = z
            (request.state.pose.orientation.x, request.state.pose.orientation.y,
             request.state.pose.orientation.z,
             request.state.pose.orientation.w) = yaw_to_quaternion(world_yaw)
            response = self._call(self._set_state, request)
            if response is not None and not response.success:
                raise RuntimeError(
                    f'set_entity_state refused to move "{self._entity_name}": '
                    f'{response.status_message}')
        finally:
            self._call(self._unpause, Empty.Request())

        if self._reseed:
            self._reseed_localization(x, y, yaw)

    def _reseed_localization(self, x: float, y: float, yaw: float):
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.pose.position.x = x
        pose.pose.pose.position.y = y
        (pose.pose.pose.orientation.x, pose.pose.pose.orientation.y,
         pose.pose.pose.orientation.z,
         pose.pose.pose.orientation.w) = yaw_to_quaternion(yaw)
        # Small but non-zero: AMCL spreads its particles from this and a zero
        # covariance collapses them onto one point.
        pose.pose.covariance[0] = 0.05
        pose.pose.covariance[7] = 0.05
        pose.pose.covariance[35] = 0.02

        # EKF first. It restarts its filter at this pose and stops publishing
        # the old odom -> base_footprint; AMCL then localises on top of an
        # odometry that already agrees with where the body is.
        request = SetPose.Request()
        request.pose = pose
        self._call(self._set_pose, request)
        # AMCL takes the map -> odom seed on a topic, not a service.
        self._initial_pose_pub.publish(pose)
