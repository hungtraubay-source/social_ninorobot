"""Launch the optional per-ByteTrack state VLM node independently of Block B."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Pass the selected YAML unchanged so model paths stay outside source code."""
    return [Node(
        package='social_perception',
        executable='social_vlm_interaction.py',
        name='social_vlm_interaction',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )]


def generate_launch_description():
    """Expose a standalone launch entry for VLM state testing without restarting B."""
    default_config = PathJoinSubstitution([
        FindPackageShare('social_perception'), 'config',
        'social_vlm_interaction.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file', default_value=default_config,
            description='VLM per-track state parameters, including local Qwen paths'),
        OpaqueFunction(function=launch_setup),
    ])
