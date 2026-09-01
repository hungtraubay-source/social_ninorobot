"""Insert the people-aware velocity limiter between Nav2 and the robot base.

Nav2's velocity_smoother publishes the final command on /cmd_vel. This filter
sits after it, so the last word on how fast the robot may approach a person is
always made with the current /people estimate.

Gazebo (default arguments)
    Nav2 -> /cmd_vel -> filter -> /cmd_vel_safe -> gazebo_ros_diff_drive
    The diff-drive plugin already listens on /cmd_vel_safe, see
    linorobot2_description/urdf/controllers/skid_steer.urdf.xacro.

        ros2 launch social_navigation social_safety.launch.py sim:=true

Real robot
    micro-ROS firmware listens on the fixed topic /cmd_vel, so the filter must
    own that name and Nav2 must be pushed off it:

    Nav2 -> /cmd_vel_nav_filtered -> filter -> /cmd_vel -> micro_ros_agent

        ros2 launch social_navigation social_safety.launch.py \\
            input_topic:=/cmd_vel_nav_filtered output_topic:=/cmd_vel

    Moving Nav2 off /cmd_vel additionally requires launching nav2_bringup with
    use_composition:=False inside a GroupAction that applies
    SetRemap('cmd_vel_smoothed', '/cmd_vel_nav_filtered'). See README.md.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_navigation'),
        'config',
        'social_navigation.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'sim',
            default_value='false',
            description='Set to true when running against Gazebo'),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the social_navigation ROS parameter file'),
        DeclareLaunchArgument(
            'people_topic',
            default_value='/people',
            description='Localized people published by social_perception'),
        DeclareLaunchArgument(
            'input_topic',
            default_value='/cmd_vel',
            description='Velocity command produced by Nav2'),
        DeclareLaunchArgument(
            'output_topic',
            default_value='/cmd_vel_safe',
            description='Velocity command the robot base actually consumes'),
        Node(
            package='social_navigation',
            executable='social_velocity_filter.py',
            name='social_velocity_filter',
            output='screen',
            # The parameter file carries tuning only. Deployment-specific values
            # are appended afterwards so they always win.
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    # rclpy declares use_sim_time as a bool, so the launch
                    # argument must be coerced instead of arriving as a string.
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('sim'), value_type=bool),
                    'people_topic': LaunchConfiguration('people_topic'),
                    'input_topic': LaunchConfiguration('input_topic'),
                    'output_topic': LaunchConfiguration('output_topic'),
                },
            ],
        ),
    ])
