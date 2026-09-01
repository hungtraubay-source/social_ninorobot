# social_navigation

ROS 2 Humble package for Gazebo Classic and Nav2. It provides:

- a Nav2 costmap plugin that writes social costs into global/local costmaps.
- a last-line velocity safety filter consuming localized people;
- simulation actors used to exercise social navigation.

Camera perception is intentionally owned by the separate `social_perception`
package. This package consumes its stable `/people` topic; it does not load
YOLO, depth images, or a VLM.

## Build and run

```bash
cd ~/ninorobot2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to social_navigation
source install/setup.bash
```

Four terminals, in this order. Releasing the actors is deliberately its own
command rather than part of perception: on real hardware there is nothing to
spawn, and in simulation the moment people walk in is worth keeping in your own
hands instead of tying it to model loading.

```bash
# 1. Gazebo. The default lirs_test.world is person-free.
ros2 launch linorobot2_gazebo gazebo.launch.py run_ekf:=false

# 2. Perception only: YOLO Pose + depth + person tracking. Loads no actors.
#    Wait for "SẴN SÀNG: camera + YOLO hoạt động, VLM đã tắt (enable_vlm=false)".
ros2 launch social_navigation social_bringup.launch.py rviz:=true

# 3. Put people into the scene, whenever you want them.
ros2 launch social_navigation social_sim.launch.py scenario:=talking

# 4. Nav2 with the social costmap layer. Do not pass rviz:=true here as well.
ros2 launch linorobot2_navigation navigation.launch.py sim:=true \
    map:=<path to map.yaml>
```

`scenario:=talking` (the default) releases `m_sweater` and `m_mechanic` into the
already-running Gazebo world. They stand 1.6 m apart, face each other, and loop
their `talk.dae` skeletal animations for the whole session.

`scenario:=gathering` runs the same two actors through a repeating cycle
instead: they walk in from outside the camera's view, hold the conversation,
walk back out, and stay hidden for a while before returning.

`scenario:=crossing` releases only `walker_1`. It walks from A to B across the
camera view, then teleports back to A before the next pass. This makes it
suitable for checking person heading, velocity and the predicted trajectory
without a second person in the scene.

Keep terminal 3 running while the actors should be visible; `Ctrl+C` there hides
them again without disturbing perception, so you can release the other scenario
straight afterwards without restarting perception. Add `wait_for_perception:=true`
if you would rather start terminals 2 and 3 together and let the actors hold
back until perception reports ready.

This package publishes nothing on `/people`; terminal 2 is what produces it.
On real hardware terminal 3 disappears and terminal 2 becomes
`social_bringup.launch.py sim:=false` plus the camera's pose in the map frame.

`lirs_test.world` defines each actor's pose and animation. The RGB-D camera is
mounted on the robot, 30 cm above `base_link`, and publishes `/camera/...` for
the `social_perception` node. Edit the world file to add people or change where
they stand.

The actual planning cost is part of `/global_costmap/costmap` and
`/local_costmap/costmap`.

## Topics

- `/people` (`social_perception/msg/People`)
