"""Run Block D independently from perception and Nav2."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_perception'),
        'config',
        'social_constraint_grounding.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='Block-D social constraint parameters'),
        Node(
            package='social_perception',
            executable='social_constraint_grounding.py',
            name='social_constraint_grounding',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
