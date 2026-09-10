"""Run a trained policy as a velocity controller.

sim:=true (default) loads rl_agent.yaml, sim:=false loads rl_agent_real.yaml.
The difference between the two is use_sim_time and the cmd_vel topic; nothing
about the policy itself changes.

localization:=true additionally starts map_server + AMCL from nav2_bringup so a
goal can be picked on the map. No planner, no controller, no behaviour tree:
AMCL only publishes the map -> odom edge, and the policy owns the base for the
whole route. The agent transforms a `map` goal into env.goal_frame itself, so
the checkpoint keeps driving in the frame it was trained in.

  RViz 2D Goal Pose -> /rl_goal_pose -> agent -> /cmd_vel -> safety filter
  map_server + AMCL -> TF map -> odom (nothing else)

people:=ground_truth points the agent at /social_gt/people instead of /people,
for testing a trained policy in simulation while block C does not exist yet.
The policy then reads the SAME labels it trained on -- scene_type filled, one
stable id per person -- so a bad run is the policy's fault rather than a
missing VLM's. It is not a second people source: same message type, same
subscription, same camera-cone and occlusion filtering in the bridge; only the
topic name changes, and social_rl/ground_truth.py stays out of the agent.

Simulation only, and it needs one TF edge to work. /social_gt/people is stamped
in `world`; gazebo.launch.py already publishes world -> map at the spawn pose,
EKF publishes odom -> base_footprint, and the missing map -> odom edge is
published here as identity. That identity is exactly what training assumed --
env.reseed_localization resets the EKF pose to the episode's map pose, so the
two frames coincide -- which is also why AMCL is the wrong thing to use for it:
its corrections would jitter people the trainer never saw jitter.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_config = PathJoinSubstitution(
        [FindPackageShare('social_rl'), 'config', 'rl_agent.yaml'])
    real_config = PathJoinSubstitution(
        [FindPackageShare('social_rl'), 'config', 'rl_agent_real.yaml'])
    localization_launch = PathJoinSubstitution(
        [FindPackageShare('nav2_bringup'), 'launch', 'localization_launch.py'])
    rviz_config = PathJoinSubstitution([
        FindPackageShare('linorobot2_navigation'),
        'rviz', 'linorobot2_navigation.rviz'])
    # The same params Nav2 would use, so AMCL keeps its tuned motion model and
    # its initial pose at the map origin. Only the localization block is read.
    sim_params = PathJoinSubstitution([
        FindPackageShare('linorobot2_navigation'), 'config', 'nav_sim.yaml'])
    real_params = PathJoinSubstitution([
        FindPackageShare('linorobot2_navigation'), 'config', 'navigation.yaml'])
    # Published by the animated_people_release plugin, in the `world` frame.
    # Named here rather than taken from EnvConfig because a launch file cannot
    # import the package it launches.
    ground_truth_people = '/social_gt/people'

    model_path = LaunchConfiguration('model_path')
    sim = LaunchConfiguration('sim')
    use_localization = LaunchConfiguration('localization')

    # Empty unless people:=ground_truth, and empty means "use whatever topic
    # the run was configured with", which is the robot's own /people.
    people_topic_override = PythonExpression(
        ["'", ground_truth_people, "' if '", LaunchConfiguration('people'),
         "' == 'ground_truth' else ''"])

    def agent(config, condition):
        return Node(
            package='social_rl',
            executable='rl_agent',
            name='social_rl_agent',
            output='screen',
            emulate_tty=True,
            condition=condition,
            parameters=[config, {
                'model_path': model_path,
                'goal_topic': LaunchConfiguration('goal_topic'),
                'env_config': LaunchConfiguration('env_config'),
                'people_topic_override': people_topic_override,
            }])

    def amcl(params_file, wanted_sim):
        # Two includes rather than one: the AMCL tuning for Gazebo and for the
        # real base live in different files, exactly as they do for Nav2.
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(localization_launch),
            condition=IfCondition(PythonExpression(
                ["'", use_localization, "' == 'true' and '",
                 sim, "' == '", wanted_sim, "'"])),
            launch_arguments={
                'map': LaunchConfiguration('map'),
                'params_file': params_file,
                'use_sim_time': sim,
            }.items())

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            description='Trained .zip, e.g. ~/social_rl_runs/<run>/final_model.zip'),
        DeclareLaunchArgument(
            'sim', default_value='true',
            description='true = Gazebo profile, false = real robot profile'),
        DeclareLaunchArgument(
            'localization', default_value='false',
            description='Start map_server + AMCL so goals can be given on the map'),
        DeclareLaunchArgument(
            'map', default_value=PathJoinSubstitution([
                FindPackageShare('linorobot2_navigation'),
                'maps', 'cafe_vlm.yaml']),
            description='Map loaded when localization:=true'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Open the existing RViz view to click goals'),
        DeclareLaunchArgument(
            'people', default_value='perception',
            description='perception = /people từ khối B, đúng thứ robot thật '
                        'có. ground_truth = /social_gt/people, CHỈ CHẠY ĐƯỢC '
                        'TRONG SIM: policy đọc đúng nhãn nó đã train (có '
                        'scene_type, có vùng nhóm, id ổn định), nên một lần '
                        'chạy tồi là lỗi của policy chứ không phải của khối C '
                        'còn thiếu. Tự bật thêm cạnh TF map -> odom identity, '
                        'nên đừng dùng chung với localization:=true'),
        DeclareLaunchArgument(
            'env_config', default_value='',
            description='Để trống thì lấy env_config.yaml nằm cạnh model, '
                        'tức đúng cấu hình đã train ra nó. Chỉ truyền đường '
                        'dẫn khác khi muốn đổi thứ gì đó ngoài nguồn người - '
                        'riêng việc đổi nguồn người thì dùng people:= ở trên, '
                        'đừng chép env_config ra sửa: bản chép sẽ lệch khỏi '
                        'cấu hình quan sát của chính checkpoint'),
        DeclareLaunchArgument(
            'zones', default_value='true',
            description='Vẽ K_soc ra /social_rl/social_costmap và '
                        '/social_rl/zone_markers cho RViz'),
        DeclareLaunchArgument(
            'prediction_times', default_value='[0.0]',
            description='Mốc thời gian của K_soc đem vẽ. Mặc định chỉ hiện '
                        'tại; [0.0,1.0,2.0] để soi cả dự báo'),
        DeclareLaunchArgument(
            'goal_topic', default_value='/rl_goal_pose',
            description='Where the agent listens for goals. Point the RViz '
                        '"2D Goal Pose" tool at it, or pass /goal_pose: no '
                        'Nav2 runs here, so that name is free'),

        # The one TF edge /social_gt/people needs, and only when nothing else
        # is publishing it. gazebo.launch.py already gives world -> map at the
        # spawn pose and EKF gives odom -> base_footprint; this closes the
        # chain. Identity because that is what training assumed:
        # env.reseed_localization resets the EKF pose to the episode's map
        # pose, so map and odom coincide. Refused alongside localization:=true
        # on purpose -- two publishers on one edge is a fight, not a fallback.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_identity',
            condition=IfCondition(PythonExpression(
                ["'", LaunchConfiguration('people'), "' == 'ground_truth' and "
                 "'", use_localization, "' == 'false'"])),
            arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
            parameters=[{'use_sim_time': ParameterValue(sim, value_type=bool)}]),

        amcl(sim_params, 'true'),
        amcl(real_params, 'false'),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': ParameterValue(sim, value_type=bool)}],
            condition=IfCondition(LaunchConfiguration('rviz'))),

        agent(sim_config, IfCondition(sim)),
        agent(real_config, UnlessCondition(sim)),

        # Khối D ra hình. Chạy cùng agent chứ không phải một terminal riêng:
        # nó chỉ nghe /social_rl/constraint_field và vẽ, không nằm trong vòng
        # điều khiển, nên quên bật nó chỉ có nghĩa là RViz trống - đúng cái
        # bẫy tốn thời gian nhất khi xem một run.
        #
        # LƯU Ý: agent chỉ publish constraint_field KHI ĐANG CÓ ĐÍCH. Tới đích
        # rồi thì nó ngừng, và vùng biến mất khỏi RViz. Không phải hỏng.
        Node(
            package='social_rl',
            executable='zone_markers',
            name='zone_markers',
            output='screen',
            parameters=[{
                'use_sim_time': ParameterValue(sim, value_type=bool),
                'prediction_times': LaunchConfiguration('prediction_times'),
            }],
            condition=IfCondition(LaunchConfiguration('zones'))),
    ])
