"""Release animated actors into an already-running Gazebo world.

This is its own terminal on purpose. Perception (social_bringup.launch.py) only
loads models and interprets the camera; when people enter the scene is your
call, so you can watch the pipeline react to an arrival you triggered. Ctrl+C
here removes the actors again without touching perception, and you can release
a different scenario straight afterwards.

scenario:=talking    (default) two actors stand face to face for the whole
                     session. Use it to tune the social region itself.
scenario:=gathering  the same two actors walk in from off camera, hold the
                     conversation, walk out, and repeat. Use it to check that
                     the costmap creates the region *and* clears it again.
scenario:=crossing   one actor repeatedly walks across the RGB-D camera.
scenario:=standing   one actor stays at world (0, -1.4) for a static test.

wait_for_perception:=true holds the actors back until social_perception reports
ready, for when you want to start both terminals at once anyway.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'scenario',
            default_value='talking',
            description='talking, gathering, crossing, or standing'),
        DeclareLaunchArgument(
            'wait_for_perception',
            default_value='false',
            description='Hold the actors back until social_perception reports ready'),
        DeclareLaunchArgument(
            'ready_timeout',
            default_value='600.0',
            description='Seconds to wait for readiness before releasing anyway'),
        TimerAction(
            period=1.0,
            actions=[Node(
                package='social_navigation',
                executable='release_animated_people.py',
                name='release_animated_people',
                output='screen',
                parameters=[{
                    'scenario': LaunchConfiguration('scenario'),
                    'wait_for_perception': ParameterValue(
                        LaunchConfiguration('wait_for_perception'),
                        value_type=bool),
                    'ready_timeout': ParameterValue(
                        LaunchConfiguration('ready_timeout'), value_type=float),
                }])]),
    ])
