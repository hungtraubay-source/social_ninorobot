"""People tracking for RL: YOLO and depth only, no VLM, no Nav2.

Same node and same parameter file the social-navigation pipeline uses
(social_perception/config/social_vlm_perception.yaml), with two values
overridden. Overriding rather than copying keeps one perception profile in the
workspace: a change to camera topics or YOLO settings reaches RL training
without anyone remembering to edit a second file.

    enable_vlm   false  -- the conversation model is never loaded. Startup drops
                          from tens of seconds to a couple, ~2 GiB of VRAM stays
                          free, and no step of the RL loop ever waits on an
                          inference. /people still comes out of YOLO + depth,
                          which is the only thing the policy reads.
    target_frame odom   -- people are published in the odometry frame instead
                          of map, so AMCL is not needed and Nav2 does not have
                          to run at all. The EKF publishes odom -> base_footprint
                          and that is the whole localization chain RL needs.

Actors in lirs_test.world have no collision geometry, so the lidar sees straight
through them. /people is the only source of people in the observation; without
this launch the policy is blind to them and learns to avoid furniture instead.

Pass vlm:=true to get the full pipeline back (people plus conversation regions).
The RL observation ignores the regions -- this is for running the social
navigation stack, not for training.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    config_file = LaunchConfiguration('config_file').perform(context)
    if not config_file:
        config_file = os.path.join(
            get_package_share_directory('social_perception'),
            'config', 'social_vlm_perception.yaml')
    enable_vlm = LaunchConfiguration('vlm').perform(context).lower() in (
        'true', '1', 'yes', 'on')

    return [Node(
        package='social_perception',
        executable='social_vlm_perception.py',
        # Must stay social_vlm_perception: that is the node name the parameter
        # file is keyed on, and a renamed node silently gets none of it.
        name='social_vlm_perception',
        output='screen',
        emulate_tty=True,
        parameters=[config_file, {
            'enable_vlm': enable_vlm,
            'target_frame': LaunchConfiguration('target_frame'),
        }])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'vlm', default_value='false',
            description='false = YOLO + depth only (what RL needs). '
                        'true = also load the VLM and publish /people_groups'),
        DeclareLaunchArgument(
            'target_frame', default_value='odom',
            description='Frame /people is published in. odom needs no AMCL; '
                        'use map when Nav2 is running'),
        DeclareLaunchArgument(
            'config_file', default_value='',
            description='Base parameter file; empty uses the simulation profile'),
        OpaqueFunction(function=launch_setup),
    ])
