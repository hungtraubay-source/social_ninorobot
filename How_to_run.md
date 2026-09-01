## Testing the robot

### 1. Run the micro-ROS agent.

This will allow the robot to receive Twist messages to control the robot, and publish odometry and IMU data straight from the microcontroller. Compared to Linorobot's ROS1 version, the odometry and IMU data published from the microcontroller use standard ROS2 messages and do not require any relay nodes to reconstruct the data to complete [sensor_msgs/Imu](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html) and [nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html) messages.

Run the agent for serial transport:

    ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0

Or for wifi transport:

    ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888

### 2. Drive around

Run teleop_twist_keyboard package and follow the instructions on the terminal on how to drive the robot:

    ros2 run teleop_twist_keyboard teleop_twist_keyboard 

Publish the cmd_vel in a steady stream at 20 Hz.

    ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

Stop the robot

    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"

### 3. Check the topics

Check if the odom and IMU data are published:

    ros2 topic list

Now you should see the following topics:

    /cmd_vel
    /imu/data
    /initialpose
    /odom/unfiltered
    /parameter_events
    /rosout
    /set_pose

Echo odometry data:

    ros2 topic echo /odom/unfiltered

Echo IMU data:

    ros2 topic echo /imu/data

Set pose:

    ros2 topic pub --once /set_pose geometry_msgs/msg/Pose2D "{x: 0.0, y: 0.0, theta: 0.0}"

    ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {frame_id: 'map'}, pose: {pose: {position: {x: 2.0, y: 1.5, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"

    ros2 topic pub --once /set_pose geometry_msgs/msg/PoseWithCovarianceStamped "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'odom'}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}"

View the rate at which data is published using:

    ros2 topic hz /cmd_vel

## Optimizing Pi 4

### 1. Turn off HDMI output (Saves ~20-30mA)

Bash
    # Temporarily turn off
    sudo /usr/bin/tvservice -o

    # Turn back on when needed
    sudo /usr/bin/tvservice -p

### 2. Disable the graphical interface at startup (Headless Mode)

Bash
    # Set the system to boot into Terminal mode (multi-user)
    sudo systemctl set-default multi-user.target

    # Then restart
    sudo reboot

### 3. Disable background services of the Desktop

Bash
    # Disable the update notification service (consumes CPU resources)
    sudo systemctl stop unattended-upgrades
    sudo systemctl disable unattended-upgrades

    # Disable the printer search service (if not in use)
    sudo systemctl stop cups
    sudo systemctl disable cups

### 4. Return to graphical mode

Bash
    sudo systemctl set-default graphical.target
    sudo reboot
