# Copyright (c) 2021 Juan Miguel Jimeno
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node, SetRemap
from launch.conditions import IfCondition, UnlessCondition
from nav2_common.launch import RewrittenYaml


MAP_NAME='playground' #change to the name of your own map here

def generate_launch_description():
    depth_sensor = os.getenv('LINOROBOT2_DEPTH_SENSOR', '')

    nav2_launch_path = PathJoinSubstitution(
        [FindPackageShare('nav2_bringup'), 'launch', 'bringup_launch.py']
    )

    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_navigation'), 'rviz', 'linorobot2_navigation.rviz']
    )

    default_map_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_navigation'), 'maps', f'{MAP_NAME}.yaml']
    )

    nav2_config_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_navigation'), 'config', 'navigation.yaml']
    )

    nav2_sim_config_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_navigation'), 'config', 'nav_sim.yaml']
    )

    social_replanning_bt_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_navigation'), 'behavior_trees',
         'navigate_to_pose_social_replanning.xml']
    )

    social_safety_launch_path = PathJoinSubstitution(
        [FindPackageShare('social_navigation'), 'launch', 'social_safety.launch.py']
    )

    # On hardware the micro-ROS firmware subscribes to the fixed topic /cmd_vel,
    # so the social velocity filter has to own that name and Nav2 has to be
    # pushed off it. nav2_bringup's velocity_smoother normally remaps its own
    # cmd_vel_smoothed output onto cmd_vel; renaming that one output moves the
    # whole stack behind the filter without touching any other topic:
    #
    #   controller_server -> cmd_vel_nav -> velocity_smoother
    #       -> /cmd_vel_nav_filtered -> social_velocity_filter -> /cmd_vel
    #
    # Gazebo needs none of this: its diff-drive plugin already listens on
    # /cmd_vel_safe, and gazebo.launch.py starts the filter itself.
    nav2_filtered_cmd_topic = '/cmd_vel_nav_filtered'

    # Both targets run the same social behavior tree. The path is only known at
    # launch time, so each config file carries a placeholder instead of a path
    # hardcoded to one machine.
    nav2_params = RewrittenYaml(
        source_file=nav2_config_path,
        param_rewrites={
            'default_nav_to_pose_bt_xml': social_replanning_bt_path,
        },
        convert_types=True,
    )

    nav2_sim_params = RewrittenYaml(
        source_file=nav2_sim_config_path,
        param_rewrites={
            'default_nav_to_pose_bt_xml': social_replanning_bt_path,
        },
        convert_types=True,
    )


    return LaunchDescription([
        DeclareLaunchArgument(
            name='sim', 
            default_value='false',
            description='Enable use_sime_time to true'
        ),

        DeclareLaunchArgument(
            name='rviz', 
            default_value='false',
            description='Run rviz'
        ),

       DeclareLaunchArgument(
            name='map', 
            default_value=default_map_path,
            description='Navigation map path'
        ),

        GroupAction(
            condition=UnlessCondition(LaunchConfiguration("sim")),
            actions=[
                # A global remap only reaches plain Node actions, so the
                # composed bringup has to be disabled for it to take effect.
                SetRemap(src='cmd_vel_smoothed', dst=nav2_filtered_cmd_topic),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(nav2_launch_path),
                    launch_arguments={
                        'map': LaunchConfiguration("map"),
                        'use_sim_time': LaunchConfiguration("sim"),
                        'params_file': nav2_params,
                        'use_composition': 'False',
                    }.items()
                ),
            ]
        ),

        # Last line of defence on hardware. Kept outside the group so the
        # remap above cannot reach it.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(social_safety_launch_path),
            condition=UnlessCondition(LaunchConfiguration("sim")),
            launch_arguments={
                'sim': 'false',
                'input_topic': nav2_filtered_cmd_topic,
                'output_topic': '/cmd_vel',
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav2_launch_path),
            condition=IfCondition(LaunchConfiguration("sim")),
            launch_arguments={
                'map': LaunchConfiguration("map"),
                'use_sim_time': LaunchConfiguration("sim"),
                'params_file': nav2_sim_params
            }.items()
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration("rviz")),
            parameters=[{'use_sim_time': LaunchConfiguration("sim")}]
        )
    ])
