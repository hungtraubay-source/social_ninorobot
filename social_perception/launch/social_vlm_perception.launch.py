"""Launch RGB-D person localization and keypoint social geometry."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    parameters = [LaunchConfiguration('config_file')]
    yolo_model_path = LaunchConfiguration('yolo_model_path').perform(context)
    if yolo_model_path:
        parameters.append({'yolo_model_path': yolo_model_path})
    return [Node(
        package='social_perception',
        executable='social_vlm_perception.py',
        name='social_vlm_perception',
        output='screen',
        parameters=parameters,
    )]


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('social_perception'),
        'config',
        'social_vlm_perception.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the social perception ROS parameter file'),
        DeclareLaunchArgument(
            'yolo_model_path',
            default_value='',
            description='Optional local YOLO Pose checkpoint overriding config'),
        OpaqueFunction(function=launch_setup),
    ])
