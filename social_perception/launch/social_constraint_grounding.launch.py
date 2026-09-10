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
        # Match Block-B coordinates explicitly: simulation odom, real map.
        DeclareLaunchArgument('expected_frame', default_value='odom',
                              description='Frame of Block-B positions (odom sim, map real)'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use Gazebo clock; false for real camera'),
        DeclareLaunchArgument(
            'social_state', default_value='auto',
            description=('auto, talking, standing (person-to-object ellipse), '
                         'stationary (fixed circular one-person Gaussian), or crossing'),
        ),
        # Keep the default RViz view clean: Block D always publishes MarkerArray
        # contours, while the blue OccupancyGrid heatmap is opt-in.
        DeclareLaunchArgument('publish_cost_grid', default_value='false',
                              description='Publish optional current Gaussian OccupancyGrid for RViz Map'),
        Node(
            package='social_perception',
            executable='social_constraint_grounding.py',
            name='social_constraint_grounding',
            output='screen',
            parameters=[LaunchConfiguration('config_file'), {
                'expected_frame': LaunchConfiguration('expected_frame'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'social_state': LaunchConfiguration('social_state'),
                'publish_cost_grid': LaunchConfiguration('publish_cost_grid'),
            }],
        ),
    ])
