"""Start a training run against a simulation that is already up.

This launch file starts nothing but the trainer. Gazebo, perception, the actors
and Nav2 stay in their own terminals so that a training run that dies -- or that
you stop to change a reward weight -- does not take the loaded VLM down with it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('social_rl'), 'config', 'rl_train.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config', default_value=default_config,
            description='Training YAML (env / observation / reward / train)'),
        DeclareLaunchArgument(
            'run_dir', default_value='',
            description='Output directory. Empty = train.output_root/<timestamp>'),
        DeclareLaunchArgument(
            'timesteps', default_value='0',
            description='Override train.total_timesteps. 0 keeps the YAML value'),
        DeclareLaunchArgument(
            'resume', default_value='',
            description='Continue from a saved .zip instead of starting fresh'),
        DeclareLaunchArgument(
            'render', default_value='',
            description='true opens the Gazebo 3D window on the running '
                        'gzserver. Empty keeps env.render from the YAML'),

        Node(
            package='social_rl',
            executable='train_rl',
            name='social_rl_train',
            output='screen',
            # Unbuffered so the per-episode lines and the SB3 tables appear as
            # they happen rather than in blocks minutes later.
            emulate_tty=True,
            arguments=[
                '--config', LaunchConfiguration('config'),
                '--run-dir', LaunchConfiguration('run_dir'),
                '--timesteps', LaunchConfiguration('timesteps'),
                '--resume', LaunchConfiguration('resume'),
                '--render', LaunchConfiguration('render'),
            ]),
    ])
