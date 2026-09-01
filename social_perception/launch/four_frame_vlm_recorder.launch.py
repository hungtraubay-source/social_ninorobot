"""Launch recorder for a legacy conversations JSON multi-image VLM dataset."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rgb_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('tracked_state_topic',
                              default_value='/people/tracked_state_json'),
        DeclareLaunchArgument('output_directory', default_value='recordings/four_frame_vlm'),
        DeclareLaunchArgument('sample_rate_hz', default_value='0.5'),
        DeclareLaunchArgument('max_samples', default_value='0'),
        DeclareLaunchArgument('frame_count', default_value='4'),
        DeclareLaunchArgument('frame_interval_s', default_value='0.5'),
        DeclareLaunchArgument('history_window_s', default_value='1.5'),
        DeclareLaunchArgument('sync_slop_seconds', default_value='0.15'),
        DeclareLaunchArgument('sync_buffer_seconds', default_value='5.0'),
        DeclareLaunchArgument('image_format', default_value='jpg'),
        DeclareLaunchArgument('jpeg_quality', default_value='95'),
        DeclareLaunchArgument('save_empty_frames', default_value='false'),
        DeclareLaunchArgument('dataset_filename', default_value='finetune_draft.json'),
        Node(
            package='social_perception',
            executable='four_frame_vlm_recorder.py',
            name='four_frame_vlm_recorder',
            output='screen',
            parameters=[{
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'tracked_state_topic': LaunchConfiguration('tracked_state_topic'),
                'output_directory': LaunchConfiguration('output_directory'),
                'sample_rate_hz': LaunchConfiguration('sample_rate_hz'),
                'max_samples': ParameterValue(LaunchConfiguration('max_samples'), value_type=int),
                'frame_count': ParameterValue(LaunchConfiguration('frame_count'), value_type=int),
                'frame_interval_s': LaunchConfiguration('frame_interval_s'),
                'history_window_s': LaunchConfiguration('history_window_s'),
                'sync_slop_seconds': LaunchConfiguration('sync_slop_seconds'),
                'sync_buffer_seconds': LaunchConfiguration('sync_buffer_seconds'),
                'image_format': LaunchConfiguration('image_format'),
                'jpeg_quality': LaunchConfiguration('jpeg_quality'),
                'save_empty_frames': ParameterValue(
                    LaunchConfiguration('save_empty_frames'), value_type=bool),
                'dataset_filename': LaunchConfiguration('dataset_filename'),
            }],
        ),
    ])
