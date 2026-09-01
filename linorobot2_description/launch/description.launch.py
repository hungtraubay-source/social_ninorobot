import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
# QUAN TRỌNG: Thêm import này
from launch_ros.parameter_descriptions import ParameterValue 

def generate_launch_description():
    # Kiểm tra biến môi trường để tránh lỗi NoneType
    robot_base = os.getenv('LINOROBOT2_BASE', '2wd') # Mặc định là 2wd nếu không tìm thấy

    urdf_path = PathJoinSubstitution(
        [FindPackageShare("linorobot2_description"), "urdf/robots", f"{robot_base}.urdf.xacro"]
    )

    rviz_config_path = PathJoinSubstitution(
        [FindPackageShare('linorobot2_description'), 'rviz', 'trajectory_view.rviz']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            name='urdf', 
            default_value=urdf_path,
            description='URDF path'
        ),
        
        DeclareLaunchArgument(
            name='publish_joints', 
            default_value='true',
            description='Launch joint_states_publisher'
        ),

        DeclareLaunchArgument(
            name='rviz', 
            default_value='false',
            description='Run rviz'
        ),

        DeclareLaunchArgument(
            name='use_sim_time', 
            default_value='false',
            description='Use simulation time'
        ),

        DeclareLaunchArgument(
            name='xacro_args',
            default_value='',
            description='Optional xacro assignments, for example publish_odom_tf:=true'
        ),

        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            condition=IfCondition(LaunchConfiguration("publish_joints")),
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    # SỬA LỖI TẠI ĐÂY: Bọc Command trong ParameterValue
                    'robot_description': ParameterValue(
                        Command(['xacro ', LaunchConfiguration('urdf'), ' ',
                                 LaunchConfiguration('xacro_args')]),
                        value_type=str
                    )
                }
            ]
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path],
            condition=IfCondition(LaunchConfiguration("rviz")),
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        )
    ])
