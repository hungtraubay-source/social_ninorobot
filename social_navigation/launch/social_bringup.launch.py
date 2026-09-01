"""Bring up the social perception pipeline for Gazebo or for real hardware.

This launch owns perception and nothing else: load the models, watch the RGB-D
stream, publish the results. Putting people into the scene is a separate
command (social_sim.launch.py), because on real hardware there is nothing to
spawn, and in simulation the moment people walk in is a decision worth keeping
in your own hands rather than tying it to model loading.

The pipeline is identical in both cases:

    RGB-D  ->  YOLO Pose + depth  ->  /people (every localized person)

`/people` feeds social_navigation::SocialLayer, declared in
linorobot2_navigation/config/nav_sim.yaml. The layer costs each person
individually.

Gazebo:

    ros2 launch social_navigation social_bringup.launch.py sim:=true

    The RGB-D camera is mounted on the simulated robot at 30 cm above
    base_link. Its URDF publishes the base_link -> camera frames; wait for the
    "SẴN SÀNG" line in this terminal, then release actors from another one:

    ros2 launch social_navigation social_sim.launch.py scenario:=talking

Real hardware, camera and GPU on the workstation, robot only navigating:

    ros2 launch social_navigation social_bringup.launch.py sim:=false \\
        camera_x:=-3.0 camera_y:=0.0 camera_z:=2.0 camera_pitch:=0.35

    Start the camera driver separately (it is vendor specific), then give this
    launch the camera's physical pose in the map frame. Those six numbers are
    what turns a pixel into a person standing at a map coordinate, so measure
    them against the same origin the map was built from.

Add rviz:=true to open the social view alongside the pipeline: the tracked
people, annotated camera image and costmap. It stays off by
default so the pipeline can run headless, over ssh, or next to an RViz window
that is already open. Do not also pass rviz:=true to navigation.launch.py --
two RViz windows on the same topics only cost frame rate.
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

import os


def is_true(context, name):
    return LaunchConfiguration(name).perform(context).lower() in (
        'true', '1', 'yes', 'on')


def launch_setup(context, *args, **kwargs):
    simulation = is_true(context, 'sim')

    # An explicit config_file always wins. Otherwise pick the profile matching
    # the target, so the two never have to be kept in sync by hand.
    config_file = LaunchConfiguration('config_file').perform(context)
    if not config_file:
        config_file = os.path.join(
            get_package_share_directory('social_perception'),
            'config',
            'social_vlm_perception.yaml' if simulation
            else 'social_vlm_perception_real.yaml')

    actions = []

    # In Gazebo the camera pose comes from the world file via gazebo.launch.py.
    # On hardware nothing knows where the tripod stands until we say so.
    if not simulation and is_true(context, 'camera_tf'):
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_social_camera',
            output='screen',
            arguments=[
                '--x', LaunchConfiguration('camera_x'),
                '--y', LaunchConfiguration('camera_y'),
                '--z', LaunchConfiguration('camera_z'),
                '--roll', LaunchConfiguration('camera_roll'),
                '--pitch', LaunchConfiguration('camera_pitch'),
                '--yaw', LaunchConfiguration('camera_yaw'),
                '--frame-id', LaunchConfiguration('map_frame'),
                '--child-frame-id', LaunchConfiguration('camera_frame'),
            ],
            parameters=[{'use_sim_time': False}]))

    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('social_perception'),
            '/launch/social_vlm_perception.launch.py',
        ]),
        condition=IfCondition(LaunchConfiguration('perception')),
        launch_arguments={
            'config_file': config_file,
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
        }.items()))

    # Per-ByteTrack semantic VLM is a separate process, so a slow/failed Qwen
    # model cannot stall RGB-D localization, ByteTrack, or Block-B trajectory
    # prediction. It is opt-in because local adapter paths and GPU budget are
    # machine specific; its YAML selects sim or real time automatically below.
    interaction_config = LaunchConfiguration('vlm_interaction_config').perform(context)
    if not interaction_config:
        interaction_config = os.path.join(
            get_package_share_directory('social_perception'), 'config',
            'social_vlm_interaction.yaml' if simulation
            else 'social_vlm_interaction_real.yaml')
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('social_perception'),
            '/launch/social_vlm_interaction.launch.py',
        ]),
        condition=IfCondition(LaunchConfiguration('vlm_interaction')),
        launch_arguments={'config_file': interaction_config}.items()))

    # Watching the regions is part of running the pipeline, not a separate
    # chore. The view works the same whether Nav2 is running or not: without it
    # the costmap layers stay empty and the markers still show.
    if is_true(context, 'rviz'):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('social_navigation'),
                '/launch/social_view.launch.py',
            ]),
            launch_arguments={
                'sim': LaunchConfiguration('sim'),
                'rviz_config': LaunchConfiguration('rviz_config'),
                'rviz_log_level': LaunchConfiguration('rviz_log_level'),
            }.items()))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'sim',
            default_value='true',
            description='true for Gazebo, false for a real camera on this machine'),
        DeclareLaunchArgument(
            'perception',
            default_value='true',
            description='Run YOLO Pose, depth localization, and person tracking'),
        DeclareLaunchArgument(
            'vlm_interaction',
            default_value='false',
            description='Run the independent Qwen per-ByteTrack state node after Block B'),
        DeclareLaunchArgument(
            'vlm_interaction_config',
            default_value='',
            description='Optional VLM state YAML; empty selects sim/real profile'),
        DeclareLaunchArgument(
            'config_file',
            default_value='',
            description='Override the parameter file; empty selects it from sim'),
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value='',
            description='Optional local YOLO Pose checkpoint overriding config'),
        DeclareLaunchArgument(
            'rviz',
            default_value='false',
            description='Open the social RViz view alongside the pipeline'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('social_navigation'),
                'rviz',
                'social_navigation.rviz',
            ]),
            description='RViz configuration used when rviz:=true'),
        DeclareLaunchArgument(
            'rviz_log_level',
            default_value='ERROR',
            description='RViz log level reaching this terminal; keep it quiet '
                        'unless RViz itself fails'),

        DeclareLaunchArgument(
            'camera_tf',
            default_value='true',
            description='Publish map -> camera_frame when sim:=false'),
        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='Frame the camera pose is measured in'),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='camera_link',
            description='Camera body frame published by the camera driver'),
        # Defaults mirror the simulated camera in lirs_test.world, which gives a
        # usable view of the floor without any measuring on the first run.
        DeclareLaunchArgument('camera_x', default_value='-3.0'),
        DeclareLaunchArgument('camera_y', default_value='0.0'),
        DeclareLaunchArgument('camera_z', default_value='2.0'),
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.35'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),

        OpaqueFunction(function=launch_setup),
    ])
