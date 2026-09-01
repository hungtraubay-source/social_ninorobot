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
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, SetEnvironmentVariable,
                            TimerAction)
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    use_sim_time = True

    # Ignore accidental workspace-root entries (commonly exported from
    # ~/.bashrc). Gazebo treats every direct child as a model and otherwise
    # floods the console with "Missing model.config" errors.
    gazebo_model_paths = []
    for path in os.getenv('GAZEBO_MODEL_PATH', '').split(os.pathsep):
        if not path or not os.path.isdir(path):
            continue
        try:
            is_model_root = any(
                os.path.isfile(os.path.join(path, child, 'model.config'))
                for child in os.listdir(path))
        except OSError:
            is_model_root = False
        if is_model_root:
            gazebo_model_paths.append(path)

    ekf_config_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_base"), "config", "ekf.yaml"]
    )

    world_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_gazebo"), "worlds", "lirs_test.world"]
    )

    # Models used by the animated-people plugin in lirs_test.world.
    social_models_path = PathJoinSubstitution(
        [FindPackageShare("social_navigation"), "models"]
    )

    # Keep Gazebo usable in a fresh terminal even when ~/.bashrc does not
    # define the robot base. This matches description.launch.py's default.
    robot_base = os.getenv('LINOROBOT2_BASE', '2wd')
    urdf_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_description"), "urdf/robots", f"{robot_base}.urdf.xacro"]
    )

    description_launch_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_description'), 'launch', 'description.launch.py']
    )

    # The simulated diff-drive plugin only subscribes to /cmd_vel_safe, so this
    # filter is the single bridge from /cmd_vel to the wheels. Starting it here
    # keeps plain teleop and SLAM working without an extra terminal; it passes
    # commands through untouched whenever /people is absent.
    social_safety_launch_path = PathJoinSubstitution(
        [FindPackageShare('social_navigation'), 'launch', 'social_safety.launch.py']
    )

    rviz_config_path = PathJoinSubstitution(
            [FindPackageShare("linorobot2_gazebo"), "rviz", "trajectory_view.rviz"]
    )

    return LaunchDescription([
        SetEnvironmentVariable(
            name='GAZEBO_MODEL_PATH',
            value=[os.pathsep.join(gazebo_model_paths), os.pathsep,
                   social_models_path]
        ),

        DeclareLaunchArgument(
            name='paused', 
            default_value='false',
            description='Start Gazebo paused'
        ),

        DeclareLaunchArgument(
            name='rviz', 
            default_value='false', # Mặc định là bật, đổi thành 'false' nếu muốn mặc định tắt
            description='Launch RViz'
        ),

        DeclareLaunchArgument(
            name='run_ekf',
            default_value='true',
            description='Run EKF localization'
        ),

        DeclareLaunchArgument(
            name='publish_odom_tf',
            default_value='false',
            description='Let Gazebo diff-drive publish odom -> base_footprint; use only when run_ekf=false'
        ),

        DeclareLaunchArgument(
            name='social_safety',
            default_value='true',
            description='Bridge /cmd_vel to /cmd_vel_safe and limit speed near people'
        ),

        DeclareLaunchArgument(
            name='urdf', 
            default_value=urdf_path,
            description='URDF path'
        ),

        DeclareLaunchArgument(
            name='odom_topic', 
            default_value='/odom',
            description='EKF out odometry topic'
        ),

        DeclareLaunchArgument(
            name='world', 
            default_value=world_path,
            description='Gazebo world'
        ),

        DeclareLaunchArgument(
            name='spawn_x', 
            default_value='-3.0',
            description='Robot spawn position in X axis'
        ),

        DeclareLaunchArgument(
            name='spawn_y', 
            default_value='0.0',
            description='Robot spawn position in Y axis'
        ),

        DeclareLaunchArgument(
            name='spawn_z', 
            # The cafe floor top is at about z=0.19. Spawn clear of it and let
            # physics settle the wheels onto the floor.
            default_value='0.35',
            description='Robot spawn position in Z axis'
        ),
            
        DeclareLaunchArgument(
            name='spawn_yaw', 
            default_value='0.0',
            description='Robot spawn heading'
        ),

        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so',  '-s', 'libgazebo_ros_init.so', LaunchConfiguration('world')],
            output='screen'
        ),

        # Gazebo and robot_state_publisher start in parallel. Wait until the
        # world and robot_description are ready before inserting the robot.
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='gazebo_ros',
                    executable='spawn_entity.py',
                    name='urdf_spawner',
                    output='screen',
                    arguments=[
                        '-topic', 'robot_description',
                        '-entity', 'linorobot2',
                        '-timeout', '30',
                        '-x', LaunchConfiguration('spawn_x'),
                        '-y', LaunchConfiguration('spawn_y'),
                        '-z', LaunchConfiguration('spawn_z'),
                        '-Y', LaunchConfiguration('spawn_yaw'),
                    ]
                )
            ]
        ),

        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    condition=IfCondition(LaunchConfiguration('paused')),
                    cmd=['ros2', 'service', 'call', '/pause_physics', 'std_srvs/srv/Empty', '{}'],
                    output='screen'
                )
            ]
        ),

        # The saved map is built by SLAM, whose origin is wherever the robot
        # started, not the Gazebo world origin. Only z stays 0: the map is 2D
        # and the navigation layers ignore height.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_map',
            arguments=[
                '--x', LaunchConfiguration('spawn_x'),
                '--y', LaunchConfiguration('spawn_y'),
                '--z', '0.0',
                '--roll', '0.0', '--pitch', '0.0',
                '--yaw', LaunchConfiguration('spawn_yaw'),
                '--frame-id', 'world',
                '--child-frame-id', 'map'
            ],
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration("rviz")), # <--- QUAN TRỌNG NHẤT
            parameters=[{'use_sim_time': use_sim_time}]
        ),

        # Node(
        #     package='linorobot2_gazebo',
        #     executable='command_timeout.py',
        #     name='command_timeout'
        # ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('run_ekf')),
            parameters=[
                {'use_sim_time': use_sim_time}, 
                ekf_config_path
            ],
            remappings=[("odometry/filtered", LaunchConfiguration("odom_topic"))]
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(description_launch_path),
            launch_arguments={
                'rviz': 'false',
                'use_sim_time': str(use_sim_time),
                'publish_joints': 'false',
                'xacro_args': ['publish_odom_tf:=',
                               LaunchConfiguration('publish_odom_tf')],
                'urdf': LaunchConfiguration('urdf')
            }.items()
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(social_safety_launch_path),
            condition=IfCondition(LaunchConfiguration('social_safety')),
            launch_arguments={'sim': 'true'}.items()
        )
    ])

#sources: 
#https://navigation.ros.org/setup_guides/index.html#
#https://answers.ros.org/question/374976/ros2-launch-gazebolaunchpy-from-my-own-launch-file/
#https://github.com/ros2/rclcpp/issues/940
