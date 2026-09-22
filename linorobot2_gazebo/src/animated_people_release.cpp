#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gazebo/common/Console.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Actor.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <social_perception/msg/people.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>

namespace linorobot2_gazebo
{
class AnimatedPeopleRelease final : public gazebo::WorldPlugin
{
public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    node_ = gazebo_ros::Node::Get(sdf);

    // Phase lengths of the gathering scenario, tunable from the world file so
    // the cycle can be matched to how long the VLM actually needs to answer.
    approach_duration_ = sdf->Get<double>("approach_duration", 8.0).first;
    talk_duration_ = sdf->Get<double>("talk_duration", 60.0).first;
    disperse_duration_ = sdf->Get<double>("disperse_duration", 8.0).first;
    away_duration_ = sdf->Get<double>("away_duration", 20.0).first;

    // World z the actors' feet should rest on. The placement constants below
    // add this to each mesh's own root height. 0.20 is the value the cafe was
    // tuned against (custom_wood_floor top at z=0); measured 10-09-2026 the
    // toe bones then sit at z~0.195. bookstore.world sets it to 0.0 so the feet
    // meet the retail floor, whose visible surface is at z~0.
    actor_ground_z_ = sdf->Get<double>("actor_ground_z", 0.20).first;

    // How far apart the two people stand while talking, centre to centre.
    //
    // This decides whether the robot can physically drive between them, which
    // is what makes the social region testable: with a gap it cannot fit
    // through, a run proves nothing about the region, because geometry alone
    // already turned the robot away.
    //
    // Measured on the global costmap rather than assumed. Each person blocks a
    // 0.635 m radius (0.45 m lethal o-space grown by the inflation layer) and
    // the robot's padded radius is 0.29 m, so the band the robot centre may
    // occupy is (gap - 1.85) m wide. Perception also reads the pair about
    // 0.24 m closer than they stand, because the bounding boxes lean inwards
    // when they gesture at each other. 2.2 m left a 0.11 m band -- two grid
    // cells, which the planner cannot be relied on to thread. This leaves
    // about 0.7 m while the perceived gap stays under maximum_talking_distance,
    // so the pair is still offered to the VLM as a conversation.
    pair_separation_ = sdf->Get<double>("pair_separation", 2.8).first;

    // Ground truth for RL training (block B and block C of the architecture,
    // replaced by the simulator during training). Publishing it from here
    // rather than reading /model_states is not a shortcut, it is the only way
    // to get two of the three fields:
    //
    //   velocity    /model_states reports zero for every actor. This plugin
    //               drives them with SetWorldPose(..., false, false), so
    //               Gazebo never computes a velocity for them. Measured
    //               28-08-2026: twist.linear is (0, 0, 0) for m_sweater while
    //               it is visibly walking.
    //   scene_type  nothing in Gazebo knows it. This plugin does -- it is the
    //               scenario it was asked to play.
    //   facing      the direction the person is looking, which is not the mesh
    //               yaw: the skinned meshes carry their own root rotation.
    //               The scenario knows the intended facing exactly.
    //
    // Frame is `world`. social_rl converts to the robot frame using the robot
    // pose from /model_states, so no TF and no localizer is involved.
    people_publisher_ = node_->create_publisher<social_perception::msg::People>(
      "/social_gt/people", rclcpp::QoS(10));
    publish_period_ = 1.0 / std::max(1.0, sdf->Get<double>(
        "ground_truth_rate", 20.0).first);

    // Trajectory prediction and robot-relative motion were dropped 10-09-2026:
    // block B is tracking only now, and block D (the Gaussian grounding node)
    // works off the current pose and velocity alone. The plugin publishes just
    // id, pose and velocity per person to match the trimmed social_perception
    // Person.msg.

    route_start_x_ = sdf->Get<double>("route_start_x", route_start_x_).first;
    route_start_y_ = sdf->Get<double>("route_start_y", route_start_y_).first;
    route_end_x_ = sdf->Get<double>("route_end_x", route_end_x_).first;
    route_end_y_ = sdf->Get<double>("route_end_y", route_end_y_).first;

    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      [this](const gazebo::common::UpdateInfo &) { UpdateActors(); });
    release_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/release", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { SpawnPeople(); });
    gather_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/gather", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { SpawnGatheringPeople(); });
    hide_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/hide", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { RemovePeople(); });
    // One entry point for RL: "<scenario>" or "<scenario> <seed>". Every
    // episode reset sends one of these, so the layout of the people changes
    // from episode to episode without the trainer having to know how an actor
    // is driven. `none` clears the scene.
    scenario_subscription_ = node_->create_subscription<std_msgs::msg::String>(
      "/animated_people/scenario", rclcpp::QoS(10),
      [this](std_msgs::msg::String::ConstSharedPtr message) {
        SelectScenario(message->data);
      });
    // 12-09-2026: /animated_people/scenario is a one-shot command, gone the
    // instant it is sent -- a subscriber that starts (or restarts, as
    // zone_markers.py does every debug session) after that message is lost
    // has no way to learn the current scenario from that topic itself.
    // TRANSIENT_LOCAL + depth 1 here means a late subscriber gets the last
    // bare scenario name immediately on connecting, no resend needed. This
    // plugin lives as long as Gazebo does, so it is the one stable place to
    // hold that state.
    scenario_state_publisher_ = node_->create_publisher<std_msgs::msg::String>(
      "/animated_people/current_scenario",
      rclcpp::QoS(1).transient_local());
    gzmsg << "Animated people factory plugin loaded\n";
  }

private:
  struct WalkingWaypoint
  {
    double time;
    double x;
    double y;
    double yaw;
  };

  struct WalkingActor
  {
    std::string name;
    std::vector<WalkingWaypoint> waypoints;
    double z;
    double animation_duration;
    double root_roll;
    // What block C would report about this person. Set by whichever scenario
    // spawned them; the geometry alone cannot distinguish "crossing" from
    // "approaching" without knowing where the robot is going.
    std::string scene_type{"walking"};
    // Whether the route loops. Crossing and approaching walk their line once
    // and then park off the scene, so a single episode sees one pass rather
    // than the same person sweeping back and forth every few seconds.
    bool loop{true};
    double start_time{0.0};
    gazebo::physics::ActorPtr actor;
  };

  struct TalkingActor
  {
    std::string name;
    double x;
    double y;
    double z;
    double yaw;
    double root_roll;
    double animation_duration;
    double phase_offset;
    // Direction this person is looking, in world coordinates. NOT `yaw`:
    // `yaw` is what goes into SetWorldPose and carries the mesh's own root
    // rotation, which differs per .dae. Measured on the known-good talking
    // pair, mesh yaw = facing + pi/2.
    double facing_yaw{0.0};
    std::string scene_type{"talking"};
    double start_time{0.0};
    gazebo::physics::ActorPtr actor;
  };

  // Each animation carries its own skeleton root offset. Reusing one set of
  // numbers for both clips makes the actor sink into or float above the floor
  // the moment it switches. Values are the first Hips keyframe of each .dae.
  struct GatheringAnimation
  {
    std::string type;
    double duration;
    double root_roll;
    double root_height;
  };

  // Walks in from off camera, holds a face-to-face conversation, then walks
  // back out and disappears. The cycle repeats, so one run exercises both the
  // appearance and the removal of a social region in the costmap.
  struct GatheringActor
  {
    std::string name;
    double start_x;
    double start_y;
    double meet_x;
    double meet_y;
    double meet_yaw;
    double exit_x;
    double exit_y;
    GatheringAnimation walk;
    GatheringAnimation talk;
    double talk_phase_offset;
    double start_time{0.0};
    int active_phase{-1};
    gazebo::physics::ActorPtr actor;
  };

  // Mesh yaw of a standing actor for a given world-frame facing direction.
  // Derived from the talking pair that was tuned by eye: m_sweater stands at
  // y = +1.4 with mesh yaw 0 and looks at m_mechanic at y = -1.4, so it faces
  // -pi/2 while its mesh yaw is 0.
  static constexpr double kStandingMeshYawOffset = M_PI / 2.0;

  void SpawnTalkingActor(
    const std::string & name,
    double x, double y, double yaw,
    double root_roll, double root_height,
    double animation_duration, double phase_offset,
    double facing_yaw = 0.0,
    const std::string & scene_type = "talking")
  {
    talking_actors_.push_back({
      name, x, y, actor_ground_z_ + root_height, yaw, root_roll,
      animation_duration, phase_offset, facing_yaw, scene_type});
  }

  void SpawnWalkingActor(
    const std::string & actor_name,
    const std::vector<WalkingWaypoint> & waypoints,
    const std::string & scene_type = "walking",
    bool loop = true,
    double z = -1.0)
  {
    if (z < 0.0) {
      z = actor_ground_z_;
    }
    if (waypoints.size() < 2 || waypoints.front().time != 0.0) {
      gzerr << "Walking actor [" << actor_name <<
        "] needs at least two waypoints and must start at time 0\n";
      return;
    }
    for (std::size_t index = 1; index < waypoints.size(); ++index) {
      if (waypoints[index].time <= waypoints[index - 1].time) {
        gzerr << "Walking actor [" << actor_name <<
          "] waypoint times must be strictly increasing\n";
        return;
      }
    }

    // m_doctor/walk.dae ends at 1.416667 s. Route time and animation time are
    // independent: the body follows the 20 s route while the gait loops.
    walking_actors_.push_back(
      {actor_name, waypoints, z, 1.416667, 1.206206, scene_type, loop});
  }

  // Height of the proxy cylinder's centre: half of its 0.90 m length, so it
  // stands on the floor and covers the laser at z 0.195 m at shin height.
  static constexpr double kProxyCentre = 0.45;
  // Same parking spot the actors use when they are not in the scene.
  static constexpr double kProxyHidden = -20.0;

  // Drag one actor's physics twin. A Gazebo Classic <actor> takes no part in
  // ray-casting or contact whatever <collision> it declares -- measured
  // 27-08-2026, both by adding a cylinder to the actor and by switching this
  // plugin's SetWorldPose to notify=true, and neither shortened a single beam.
  // A plain <model> in the same spot did (2.903 m against 2.910 m expected),
  // so lirs_test.world carries one <name>_proxy model per actor and this keeps
  // it under the actor's feet.
  void MoveProxy(const std::string & actor_name, double x, double y, double z)
  {
    const std::string proxy_name = actor_name + "_proxy";
    const auto model = world_->ModelByName(proxy_name);
    if (!model) {
      // Warn once per name rather than every world update: a world without
      // the proxies still runs, the lidar just cannot see people in it.
      if (missing_proxies_.insert(proxy_name).second) {
        gzwarn << "No model [" << proxy_name << "] in this world; the lidar "
          "will not see actor [" << actor_name << "]\n";
      }
      return;
    }
    model->SetWorldPose(ignition::math::Pose3d(x, y, z, 0.0, 0.0, 0.0));
  }

  static double InterpolateYaw(double from, double to, double ratio)
  {
    const double difference = std::atan2(std::sin(to - from), std::cos(to - from));
    return from + ratio * difference;
  }

  // One person as block B and block C would describe them, in world
  // coordinates. Accumulated fresh on every world update and handed to
  // PublishPeople.
  struct PersonState
  {
    std::string name;
    double x;
    double y;
    double facing_yaw;
    std::string scene_type;
  };

  void UpdateActors()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    const double now = world_->SimTime().Double();
    if (!spawned_) {
      // An empty list is information, not silence: it is how the trainer sees
      // that the previous episode's people are gone rather than that the
      // publisher died.
      pending_people_.clear();
      PublishPeople(now);
      return;
    }
    pending_people_.clear();

    // Custom trajectories let the plugin control animation time independently
    // from the actors' fixed world poses.
    for (auto & talking : talking_actors_) {
      if (!talking.actor) {
        talking.actor = boost::dynamic_pointer_cast<gazebo::physics::Actor>(
          world_->ModelByName(talking.name));
        if (!talking.actor) {
          continue;
        }

        // With a custom trajectory Gazebo does not replace ScriptTime with
        // its internal clock in Actor::Update().
        auto trajectory = std::make_shared<gazebo::physics::TrajectoryInfo>();
        trajectory->id = 0;
        trajectory->type = "talk";
        trajectory->duration = talking.animation_duration;
        trajectory->startTime = 0.0;
        trajectory->endTime = talking.animation_duration;
        trajectory->translated = false;
        talking.actor->SetCustomTrajectory(trajectory);
        talking.actor->SetWorldPose(
          ignition::math::Pose3d(
            talking.x, talking.y, talking.z,
            talking.root_roll, 0.0, talking.yaw),
          false, false);
        MoveProxy(talking.name, talking.x, talking.y, kProxyCentre);
        talking.start_time = now;
        talking.actor->Play();
        gzmsg << "Talking actor [" << talking.name << "] started\n";
      }

      const double elapsed = std::max(0.0, now - talking.start_time);
      const double animation_time = std::fmod(
        elapsed + talking.phase_offset, talking.animation_duration);
      talking.actor->SetScriptTime(animation_time);
      pending_people_.push_back(
        {talking.name, talking.x, talking.y, talking.facing_yaw,
          talking.scene_type});
    }

    for (auto & walking : walking_actors_) {
      if (!walking.actor) {
        walking.actor = boost::dynamic_pointer_cast<gazebo::physics::Actor>(
          world_->ModelByName(walking.name));
        if (!walking.actor) {
          continue;
        }
        walking.start_time = now;
        auto trajectory = std::make_shared<gazebo::physics::TrajectoryInfo>();
        trajectory->id = 0;
        trajectory->type = "walking";
        trajectory->duration = walking.animation_duration;
        trajectory->startTime = 0.0;
        trajectory->endTime = walking.animation_duration;
        trajectory->translated = false;
        walking.actor->SetCustomTrajectory(trajectory);
        walking.actor->Play();
        gzmsg << "Walking actor [" << walking.name << "] started\n";
      }

      const double duration = walking.waypoints.back().time;
      const double travelled = std::max(0.0, now - walking.start_time);
      if (!walking.loop && travelled > duration) {
        // One pass only. Park under the floor rather than freezing at the last
        // waypoint: a person standing motionless at the edge of the scene for
        // the rest of the episode is a constraint the policy would have to
        // keep paying attention to, and it is not the situation being taught.
        walking.actor->SetScriptTime(0.0);
        walking.actor->SetWorldPose(
          ignition::math::Pose3d(
            walking.waypoints.back().x, walking.waypoints.back().y, -20.0,
            0.0, 0.0, 0.0), false, false);
        MoveProxy(walking.name, walking.waypoints.back().x,
          walking.waypoints.back().y, kProxyHidden);
        continue;
      }
      const double route_time = walking.loop ?
        std::fmod(travelled, duration) : std::min(travelled, duration);
      auto next = std::upper_bound(
        walking.waypoints.begin(), walking.waypoints.end(), route_time,
        [](double time, const WalkingWaypoint & waypoint) {return time < waypoint.time;});
      if (next == walking.waypoints.begin()) {
        ++next;
      }
      if (next == walking.waypoints.end()) {
        next = walking.waypoints.end() - 1;
      }
      const auto & previous = *(next - 1);
      const double ratio = (route_time - previous.time) / (next->time - previous.time);
      const double x = previous.x + ratio * (next->x - previous.x);
      const double y = previous.y + ratio * (next->y - previous.y);
      const double yaw = InterpolateYaw(previous.yaw, next->yaw, ratio);

      // The existing walk.dae drives the skeleton while the world plugin
      // drives the route explicitly.
      walking.actor->SetScriptTime(std::fmod(route_time, walking.animation_duration));
      walking.actor->SetWorldPose(
        ignition::math::Pose3d(
          x, y, walking.z + 0.886948, walking.root_roll, 0.0,
          // + kStandingMeshYawOffset, 28-08-2026. This line used to pass `yaw`
          // straight through on the claim that "for this actor the mesh yaw
          // already is the travel direction". It is not: the walker was
          // rendered 90 degrees off its own path.
          //
          // The offset does not come from the animation. Measured on the first
          // Hips keyframe of every .dae in use -- m_doctor/walk.dae -1.45 deg,
          // m_doctor/talk.dae -0.21 deg, m_mechanic/talk.dae +0.22 deg -- none
          // of them bakes in a quarter turn. (Same parse reproduces root_roll
          // 1.205901 against the 1.206206 in this file and z 0.886947 against
          // 0.886948, so the numbers are the right ones.) It comes from the
          // character's REST POSE, which talk.dae and walk.dae of one model
          // share. So the +pi/2 already proven on the standing pair applies to
          // every Male/*.dae here, walking included, and the constant's name
          // is now narrower than its meaning.
          //
          // Sign confirmed by eye in Gazebo, 28-08-2026: the walker now faces
          // along its own path. That check needed a human -- +pi/2 and -pi/2
          // are both "90 degrees off" and no measurement here separates them.
          yaw + kStandingMeshYawOffset), false, false);
      MoveProxy(walking.name, x, y, kProxyCentre);
      // Ground truth keeps publishing the TRAVEL direction, unchanged and
      // without the offset: `facing` is where the person is looking, and the
      // mesh yaw is an artefact of how the asset was authored. Measured before
      // this fix -- published facing matched atan2(vy, vx) to +0.0 degrees over
      // 84 samples -- so the data was already right and only the rendering was
      // wrong. Do not "fix" this line to match the one above.
      pending_people_.push_back(
        {walking.name, x, y, yaw, walking.scene_type});
    }

    UpdateGatheringActors(now);
    PublishPeople(now);
  }

  void UpdateGatheringActors(double now)
  {
    const double cycle = approach_duration_ + talk_duration_ +
      disperse_duration_ + away_duration_;
    if (cycle <= 0.0) {
      return;
    }

    for (auto & person : gathering_actors_) {
      if (!person.actor) {
        person.actor = boost::dynamic_pointer_cast<gazebo::physics::Actor>(
          world_->ModelByName(person.name));
        if (!person.actor) {
          continue;
        }
        person.start_time = now;
        person.active_phase = -1;
        person.actor->Play();
        gzmsg << "Gathering actor [" << person.name << "] started\n";
      }

      const double elapsed = std::fmod(std::max(0.0, now - person.start_time), cycle);
      int phase = 0;
      double phase_time = elapsed;
      if (phase_time >= approach_duration_) {
        phase_time -= approach_duration_;
        phase = 1;
        if (phase_time >= talk_duration_) {
          phase_time -= talk_duration_;
          phase = 2;
          if (phase_time >= disperse_duration_) {
            phase_time -= disperse_duration_;
            phase = 3;
          }
        }
      }

      const GatheringAnimation & animation =
        (phase == 1) ? person.talk : person.walk;

      // Gazebo restarts the clip whenever a custom trajectory is installed, so
      // this must happen on a transition only, never every world update.
      if (phase != person.active_phase) {
        auto trajectory = std::make_shared<gazebo::physics::TrajectoryInfo>();
        trajectory->id = 0;
        trajectory->type = animation.type;
        trajectory->duration = animation.duration;
        trajectory->startTime = 0.0;
        trajectory->endTime = animation.duration;
        trajectory->translated = false;
        person.actor->SetCustomTrajectory(trajectory);
        person.active_phase = phase;
      }

      // Phase 3 parks the actor under the floor instead of leaving it standing
      // off camera. The social region must provably disappear from the costmap,
      // and a person the camera cannot see is the only way to prove it.
      if (phase == 3) {
        person.actor->SetScriptTime(0.0);
        person.actor->SetWorldPose(
          ignition::math::Pose3d(person.exit_x, person.exit_y, -20.0, 0.0, 0.0, 0.0),
          false, false);
        MoveProxy(person.name, person.exit_x, person.exit_y, kProxyHidden);
        continue;
      }

      double x = person.meet_x;
      double y = person.meet_y;
      double yaw = person.meet_yaw;
      if (phase == 0) {
        const double ratio = std::clamp(phase_time / approach_duration_, 0.0, 1.0);
        x = person.start_x + ratio * (person.meet_x - person.start_x);
        y = person.start_y + ratio * (person.meet_y - person.start_y);
        const double heading = std::atan2(
          person.meet_y - person.start_y, person.meet_x - person.start_x);
        // Face the direction of travel, then turn to the partner just before
        // arriving so the conversation pose is not reached with a snap.
        yaw = (ratio < 0.75) ? heading :
          InterpolateYaw(heading, person.meet_yaw, (ratio - 0.75) / 0.25);
      } else if (phase == 2) {
        const double ratio = std::clamp(phase_time / disperse_duration_, 0.0, 1.0);
        x = person.meet_x + ratio * (person.exit_x - person.meet_x);
        y = person.meet_y + ratio * (person.exit_y - person.meet_y);
        const double heading = std::atan2(
          person.exit_y - person.meet_y, person.exit_x - person.meet_x);
        yaw = (ratio < 0.25) ?
          InterpolateYaw(person.meet_yaw, heading, ratio / 0.25) : heading;
      }

      const double offset = (phase == 1) ? person.talk_phase_offset : 0.0;
      person.actor->SetScriptTime(
        std::fmod(phase_time + offset, animation.duration));
      person.actor->SetWorldPose(
        ignition::math::Pose3d(
          x, y, actor_ground_z_ + animation.root_height,
          animation.root_roll, 0.0, yaw),
        false, false);
      MoveProxy(person.name, x, y, kProxyCentre);
      // Standing still in the conversation the mesh carries the +pi/2 root
      // offset; walking in and out it points along the route.
      pending_people_.push_back(
        {person.name, x, y,
          (phase == 1) ? yaw - kStandingMeshYawOffset : yaw,
          (phase == 1) ? std::string("talking") : std::string("walking")});
    }
  }

  // ------------------------------------------------------------ ground truth

  // Velocity is differentiated here rather than taken from Gazebo. Every actor
  // is driven with SetWorldPose(..., false, false), which moves the body
  // without telling the physics engine, so Actor::WorldLinearVel and the twist
  // in /model_states both stay at zero for a person who is visibly walking
  // (measured 28-08-2026). The positions above are exact, so a difference over
  // one publish period is exact too.
  void PublishPeople(double now)
  {
    if (now - last_publish_time_ < publish_period_ && now >= last_publish_time_) {
      return;
    }
    const double interval = now - last_publish_time_;
    last_publish_time_ = now;

    social_perception::msg::People message;
    message.header.stamp.sec = static_cast<int32_t>(now);
    message.header.stamp.nanosec = static_cast<uint32_t>(
      (now - std::floor(now)) * 1e9);
    // World coordinates. social_rl converts into the robot frame using the
    // robot's own pose from /model_states, so this needs no TF and no
    // localizer -- the training loop runs without AMCL entirely.
    message.header.frame_id = "world";

    std::unordered_map<std::string, std::pair<double, double>> current;
    for (const auto & person : pending_people_) {
      social_perception::msg::Person entry;
      entry.id = person.name;
      entry.pose.position.x = person.x;
      entry.pose.position.y = person.y;
      // Ground level. Block D works in 2D and the skeleton root height would
      // only be a number nobody reads.
      entry.pose.position.z = 0.0;
      entry.pose.orientation.z = std::sin(person.facing_yaw / 2.0);
      entry.pose.orientation.w = std::cos(person.facing_yaw / 2.0);

      double vx = 0.0;
      double vy = 0.0;
      const auto previous = previous_positions_.find(person.name);
      if (previous != previous_positions_.end() && interval > 1e-6) {
        const double dx = person.x - previous->second.first;
        const double dy = person.y - previous->second.second;
        // A jump this large in one publish period is a scenario change or a
        // teleport, not walking. Reporting it as velocity would put a
        // prediction ten metres away into the constraint field.
        if (std::hypot(dx, dy) < kMaximumStep) {
          vx = dx / interval;
          vy = dy / interval;
        }
      }
      entry.velocity.linear.x = vx;
      entry.velocity.linear.y = vy;
      // scene_type is no longer carried on Person.msg. The training scenario is
      // known from /animated_people/scenario, which ground_truth.py reads
      // directly; block D on the real robot infers it from count and speed.
      message.people.push_back(entry);
      current.emplace(person.name, std::make_pair(person.x, person.y));
    }
    previous_positions_ = std::move(current);
    people_publisher_->publish(message);
  }

  // 1.0 m in one publish period is 20 m/s at the default 20 Hz -- far past
  // walking, and reached only by a teleport.
  static constexpr double kMaximumStep = 1.0;

  void SpawnPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    talking_actors_.clear();
    walking_actors_.clear();

    // The world preloads these actors below the floor. Loading them at world
    // startup is important: Gazebo classic doesn't reliably animate actors
    // inserted later with World::InsertModelString.
    // Durations come from the final keyframe in each talk.dae. A phase offset
    // keeps the two people from making their strongest gestures in lockstep.
    const double half_gap = 0.5 * pair_separation_;
    // Mesh yaw 0 and pi, facing -pi/2 and +pi/2: they look at each other
    // across the y axis while their meshes point along x. That gap is the
    // +pi/2 root rotation the .dae files carry, and it is why the ground truth
    // publishes facing separately from the pose that goes into Gazebo.
    SpawnTalkingActor(
      "m_sweater", 0.0, half_gap, 0.0, 1.112927, 0.878344, 3.75, 0.0,
      -M_PI / 2.0, "talking");
    SpawnTalkingActor(
      "m_mechanic", 0.0, -half_gap, 3.14159265359,
      1.195040, 0.836640, 10.25, 1.25, M_PI / 2.0, "talking");

    // Example: uncomment this block to add a walking person. Only this spawn
    // block is needed; RemovePeople() records and removes it automatically.
    // SpawnWalkingActor(
    //   "walker_1",
    //   {
    //     {0.0, -2.0, -1.0, 0.0},
    //     {5.0,  2.0, -1.0, 0.0},
    //     {10.0, 2.0,  1.0, 1.57079632679},
    //     {15.0, -2.0, 1.0, 3.14159265359},
    //     {20.0, -2.0, -1.0, -1.57079632679},
    //   });
    spawned_ = true;
    gzmsg << "Spawned animated conversation pair\n";
  }

  void SpawnGatheringPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    talking_actors_.clear();
    walking_actors_.clear();
    gathering_actors_.clear();
    const double half_gap = 0.5 * pair_separation_;

    // Every constant below is the first Hips keyframe of the matching .dae:
    // duration, root roll and root height. Mixing the talk values into the walk
    // phase tilts the body and buries the feet.
    const GatheringAnimation sweater_walk{"walking", 1.208333, 1.228690, 0.927609};
    const GatheringAnimation sweater_talk{"talk", 3.750000, 1.218455, 0.903490};
    const GatheringAnimation mechanic_walk{"walking", 1.208333, 1.198338, 0.870229};
    const GatheringAnimation mechanic_talk{"talk", 10.250000, 1.195040, 0.836640};

    // They meet at the same spot the static pair uses, which is centred in the
    // dataset camera's field of view. The start and exit points sit outside
    // that view, so the camera sees an empty scene between conversations.
    gathering_actors_.push_back({
      "m_sweater",
      0.0, 3.5,             // start, off camera
      0.0, half_gap, 0.0,   // conversation pose
      0.0, 3.5,             // exit, back the way they came
      sweater_walk, sweater_talk, 0.0});
    gathering_actors_.push_back({
      "m_mechanic",
      0.0, -3.5,
      0.0, -half_gap, 3.14159265359,
      0.0, -3.5,
      mechanic_walk, mechanic_talk, 1.25});

    spawned_ = true;
    gzmsg << "Spawned gathering scenario: approach " << approach_duration_ <<
      "s, talk " << talk_duration_ << "s, disperse " << disperse_duration_ <<
      "s, away " << away_duration_ << "s\n";
  }

  // ---------------------------------------------------------- RL scenarios
  //
  // Four situations, one per scene_type the constraint field distinguishes.
  // They are randomised on every call so that an episode reset changes the
  // layout without the trainer knowing how an actor is driven -- it sends one
  // string and gets a fresh scene.
  //
  // All of them are laid out around the nominal route the robot is learning to
  // drive, in world coordinates. Placing people at fixed world points instead
  // would put most of them nowhere near the robot, and an episode where nobody
  // is ever in the way teaches nothing about avoidance.

  double NextUniform(double low, double high)
  {
    return std::uniform_real_distribution<double>(low, high)(rng_);
  }

  // A point on the nominal route, `ratio` of the way from start to end, pushed
  // sideways by `offset` metres (positive = left of the direction of travel).
  void RoutePoint(double ratio, double offset, double * x, double * y) const
  {
    const double heading = std::atan2(
      route_end_y_ - route_start_y_, route_end_x_ - route_start_x_);
    *x = route_start_x_ + ratio * (route_end_x_ - route_start_x_)
      - offset * std::sin(heading);
    *y = route_start_y_ + ratio * (route_end_y_ - route_start_y_)
      + offset * std::cos(heading);
  }

  double RouteHeading() const
  {
    return std::atan2(
      route_end_y_ - route_start_y_, route_end_x_ - route_start_x_);
  }

  // Two waypoints far enough apart that the walker is still moving when the
  // episode ends, at a walking pace.
  void WalkLine(
    const std::string & actor_name, const std::string & scene_type,
    double from_x, double from_y, double to_x, double to_y, double speed)
  {
    const double heading = std::atan2(to_y - from_y, to_x - from_x);
    const double duration = std::hypot(to_x - from_x, to_y - from_y) /
      std::max(0.1, speed);
    SpawnWalkingActor(
      actor_name, {{0.0, from_x, from_y, heading},
        {duration, to_x, to_y, heading}}, scene_type, false);
  }

  // Somebody walks across the route, left to right or right to left. The
  // robot has to decide between yielding and passing behind.
  void SpawnCrossingPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    ClearScenario();
    const double heading = RouteHeading();
    const double sideways = heading + M_PI / 2.0;
    // Đúng MỘT người, 28-08-2026. Trước đó là
    //   (NextUniform(0.0, 1.0) < 0.35) ? 2 : 1
    // tức 35% số tập có hai người cắt ngang; đo 40 lần thả ra 70/30, khớp.
    // Đổi theo yêu cầu để tình huống cắt ngang chỉ có một người.
    //
    // Đánh đổi: policy không còn tập nào phải chọn khe giữa HAI người cùng
    // cắt ngang, mà đó là tình huống duy nhất trong `crossing` bắt nó đọc
    // trường ràng buộc thay vì chỉ né vật gần nhất. Đưa lại chỉ là khôi phục
    // một dòng. `approaching` vẫn giữ 30% hai người - không đụng tới.
    const int count = 1;
    static const char * kWalkers[] = {"walker_1", "walker_2"};
    for (int index = 0; index < count; ++index) {
      // Where along the route they cross, and how far out they start. The
      // robot covers the route in about 10 s, so a crossing placed between
      // 0.35 and 0.75 of the way along is one it actually meets.
      const double ratio = NextUniform(0.35, 0.75);
      const double reach = NextUniform(2.5, 3.5);
      const double side = (NextUniform(0.0, 1.0) < 0.5) ? 1.0 : -1.0;
      double centre_x = 0.0;
      double centre_y = 0.0;
      RoutePoint(ratio, 0.0, &centre_x, &centre_y);
      WalkLine(
        // scene_type "passing": crossing and approaching were merged into one
        // label 10-09-2026. The two spawn geometries stay separate so a manual
        // `crossing` / `approaching` command still lays out the shape it names,
        // but the ground truth reports the single situation the policy sees.
        kWalkers[index], "passing",
        centre_x + side * reach * std::cos(sideways),
        centre_y + side * reach * std::sin(sideways),
        centre_x - side * reach * std::cos(sideways),
        centre_y - side * reach * std::sin(sideways),
        // 0.8-1.0 (11-09-2026, requested range). Was 0.40-0.75, tuned so the
        // walker's average speed (reach/t) matched the robot reaching a
        // crossing placed anywhere in ratio 0.35-0.75: 0.75-1.04 m/s for the
        // earliest placement but only 0.35-0.66 m/s mid/late-route -- see the
        // arithmetic this replaced, still true, just above in git blame. At a
        // FIXED 0.8-1.0 m/s the walker crosses too early relative to the
        // robot for roughly the back half of that ratio range, which is the
        // "empty crossing" failure this range used to avoid. Not re-tuned
        // here; verify with `ros2 topic echo /social_gt/people` that the
        // walker and robot are still both present near the crossing point
        // before trusting this in a training run.
        NextUniform(0.8, 1.0));
    }
    spawned_ = true;
    gzmsg << "Spawned crossing scenario with " << count << " walker(s)\n";
  }

  // Somebody walks straight at the robot. Head on, so both are closing at the
  // sum of their speeds and the robot has the least time to react.
  void SpawnApproachingPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    ClearScenario();
    // Đúng MỘT người, 28-08-2026, cùng lý do như SpawnCrossingPeople. Trước
    // đó là (NextUniform(0.0, 1.0) < 0.3) ? 2 : 1.
    const int count = 1;
    static const char * kWalkers[] = {"walker_1", "walker_2"};
    for (int index = 0; index < count; ++index) {
      // Offset sideways so the two do not walk the same line, and so the robot
      // does not always meet a person dead centre.
      const double offset = NextUniform(-0.8, 0.8) + index * 1.0;
      double from_x = 0.0;
      double from_y = 0.0;
      double to_x = 0.0;
      double to_y = 0.0;
      RoutePoint(NextUniform(0.95, 1.15), offset, &from_x, &from_y);
      RoutePoint(NextUniform(-0.25, -0.05), offset, &to_x, &to_y);
      WalkLine(
        kWalkers[index], "passing", from_x, from_y, to_x, to_y,
        // 0.8-1.0 (11-09-2026, requested range). Was 0.5-0.9.
        NextUniform(0.8, 1.0));
    }
    spawned_ = true;
    gzmsg << "Spawned approaching scenario with " << count << " walker(s)\n";
  }

  // Two people standing side by side with their backs to the robot, looking at
  // something. They are a group, but not a conversation the robot would cut
  // through: passing behind them is the socially correct move, and the
  // constraint field has to say so or the policy learns to treat every group
  // like a face-to-face pair.
  void SpawnBacksTurnedPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    ClearScenario();
    // Facing away from the robot means facing along the route, give or take.
    const double facing = RouteHeading() + NextUniform(-0.4, 0.4);
    const double half_gap = 0.5 * NextUniform(0.7, 1.1);
    double centre_x = 0.0;
    double centre_y = 0.0;
    RoutePoint(NextUniform(0.45, 0.75), NextUniform(-0.6, 0.6),
      &centre_x, &centre_y);
    // Shoulder to shoulder: the offset between them is perpendicular to where
    // they are looking.
    const double shoulder = facing + M_PI / 2.0;
    SpawnTalkingActor(
      "m_sweater",
      centre_x + half_gap * std::cos(shoulder),
      centre_y + half_gap * std::sin(shoulder),
      facing + kStandingMeshYawOffset, 1.112927, 0.878344, 3.75, 0.0,
      facing, "backs_turned");
    SpawnTalkingActor(
      "m_mechanic",
      centre_x - half_gap * std::cos(shoulder),
      centre_y - half_gap * std::sin(shoulder),
      facing + kStandingMeshYawOffset, 1.195040, 0.836640, 10.25, 1.25,
      facing, "backs_turned");
    spawned_ = true;
    gzmsg << "Spawned backs-turned pair facing " << facing << " rad\n";
  }

  // One person stands still, scene_type `waiting`, facing a real bookshelf
  // (BookshelfA_01_001 in bookstore.world) -- but placed via RoutePoint like
  // talking/backs_turned, i.e. NOT close enough to it to ever overlap the
  // mesh (11-09-2026: an earlier version stood the person right next to the
  // shelf and that put the actor inside it as often as not, since this
  // package cannot see the mesh to place against it precisely). Facing is a
  // real bearing to the shelf for visual plausibility only; the Gaussian
  // math still uses the tuned constant `waiting_distance`
  // (constraint_field.py), not this real distance, which is usually several
  // metres and would otherwise inflate the region for no reason.
  void SpawnWaitingPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    ClearScenario();
    double person_x = 0.0;
    double person_y = 0.0;
    RoutePoint(NextUniform(0.45, 0.75), NextUniform(-0.6, 0.6),
      &person_x, &person_y);
    const double shelf_x = 1.405153;
    const double shelf_y = -1.449740;
    const double facing = std::atan2(shelf_y - person_y, shelf_x - person_x);
    SpawnTalkingActor(
      "m_sweater", person_x, person_y,
      facing + kStandingMeshYawOffset, 1.112927, 0.878344, 3.75, 0.0,
      facing, "waiting");
    spawned_ = true;
    gzmsg << "Spawned waiting person at (" << person_x << ", " << person_y <<
      "), facing shelf at (" << shelf_x << ", " << shelf_y << ")\n";
  }

  // The conversation pair, but placed somewhere on the route rather than at
  // the fixed spot SpawnPeople uses. The line between them is what the robot
  // must not cross.
  void SpawnTalkingPeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    ClearScenario();
    // The axis they face each other along. Randomised so the robot does not
    // always meet the pair broadside.
    const double axis = NextUniform(-M_PI, M_PI);
    // Person-to-person gap in [1.0, 1.5] m (11-09-2026, requested range).
    // Was NextUniform(1.6, pair_separation_) = [1.6, 2.8]; narrower on purpose
    // to make the pair a tighter squeeze for the policy. No physical actor
    // collision exists to violate (the proxy collision cylinders are
    // commented out in lirs_test.world), so nothing stops the robot driving
    // this close mechanically -- it is a training-difficulty choice only.
    const double half_gap = 0.5 * NextUniform(1.0, 1.5);
    double centre_x = 0.0;
    double centre_y = 0.0;
    // OFF THE ROUTE ON PURPOSE, NOT ON IT (31-08-2026). This used to be
    // NextUniform(-0.5, 0.5), which put the pair squarely across the robot's
    // path in nearly every episode -- and that is the one geometry the policy
    // cannot learn from. Measured on the field this world produces, sweeping
    // the robot's lateral detour from -3.2 to +3.2 m and asking whether the
    // intrusion integral has a downhill run from "straight ahead" to clear:
    //
    //   pair offset   integral going straight   episodes with a usable slope
    //        0.0 m                      26.0                            6%
    //        0.4 m                      24.4                           67%
    //        0.8 m                      20.0                          100%
    //        1.2 m                       8.9                           72%
    //        1.6 m                       2.7                           28%
    //        2.0 m                       0.1                            0%
    //
    // Dead centre the profile is a plateau with a cliff: stepping a little to
    // either side costs the SAME as driving through the middle, so there is
    // nothing for a gradient to follow and the two correct answers sit either
    // side of a barrier. Past 1.6 m there is no longer anything to avoid and
    // the episode is a `none` wearing a costume. A band with a random side is
    // where every episode has both: somebody in the way, and a direction that
    // visibly pays to move in.
    //
    // NARROWED 1.4 -> 1.0 (31-08-2026, second pass). [0.4, 1.4] worked -- run
    // 20260831_160225 took the `talking` avoidance score from 8% to 73% at its
    // best checkpoint -- but it bought that with a failure mode that was not
    // there before: five of seventeen `talking` episodes in the late half
    // timed out with the robot sitting INSIDE the o-space, 0.12-0.22 m from a
    // person, for the full 250 steps, returns down to -863.
    //
    // That is a pocket, not indecision. Offset far enough and the pair sits
    // near a wall; the side with no room left is still one of the two the
    // robot may pick, and allow_reverse is false (ros_interface.py:216 -- the
    // base has no rear sensor), so a robot that picks it cannot back out and
    // has to turn around inside the gap. The upper end of the band is what
    // creates those pockets, and it is also the end that buys the least:
    // 1.2 m already drops to 72% of episodes having a usable slope and the
    // straight-line integral to 8.9, against 100% and 20.0 at 0.8 m.
    //
    // This is stage one of a curriculum. Once the policy has the direction
    // rule, widen back towards 0.0 so it also handles the pair blocking the
    // route head-on, which is the case it has to break the symmetry itself.
    //
    // 0.4-1.0 -> 0.7-0.9 (12-09-2026, requested for the none80/talking20
    // phase: "robot luôn phải gặp vùng xã hội trên đường tới đích"). The
    // measured table above says 0.8 m is the ONE point with 100% of episodes
    // having a usable slope (vs 67% at 0.4 m, 72% at 1.2 m) -- narrowing to
    // a tight band around it is the closest thing to "always" this geometry
    // supports without reverting to 0.0 m, which the same measurement showed
    // has NO gradient to learn from at all (6% usable, a flat plateau with a
    // cliff). Not re-measured at this narrower band; if `talking` episodes
    // start looking suspiciously easy or the pocket failure mode from the
    // 31-08 note above reappears, widen back towards 0.4-1.0.
    //
    // 0.7-0.9 -> 0.3-0.5 (16-09-2026, theo yêu cầu, để robot CHẮC CHẮN gặp
    // người trên đường tới đích). CẢNH BÁO: đi NGƯỢC với số đo ở trên -- bảng
    // đo chỉ có 0.0/0.4/0.8/1.2/1.6/2.0, không có điểm nào ở 0.3-0.5, nhưng
    // nội suy giữa 0.0 m (6% usable) và 0.4 m (67% usable) thì dải này nhiều
    // khả năng THẤP HƠN 67%, tức đường dốc để học né MỜ hơn hẳn so với 0.8 m
    // (100%) đang dùng trước đó.
    //
    // 0.3-0.5 -> 0.7-0.9 (17-09-2026, theo yêu cầu, TRẢ VỀ). Xác nhận đúng
    // nghi ngờ ở trên: run 20260917_003717 (resume 165352, đúng dải 0.3-0.5 +
    // tỉ lệ 40/60) cho `ep_rew_mean` giảm ròng 15.2 -> 10.9 suốt 100k bước,
    // kèm một giai đoạn sập rõ (step 208k-223k: goal 40-55%, timeout 45-50%)
    // không hồi phục hoàn toàn. Rộng lại về mức đã đo 100% "đường dốc học
    // được" trước khi bắt đầu giai đoạn học né nghiêm túc.
    //
    // 0.7-0.9 -> 0.5-0.6 (19-09-2026, theo yêu cầu). Resume từ checkpoint
    // 230352 (run 20260918_020756, train ở đúng 0.7-0.9 + 40/60 + lr 1e-4,
    // PASS 4/4 ở offset train nhưng FAIL clear_episodes ở offset=0: 43.3%/
    // 52.5%). Chỉ đổi offset so với điều kiện đã train ra 230352. Nội suy từ
    // bảng đo: ~75-85% tập có đường dốc học né (0.4 m: 67%, 0.8 m: 100%);
    // lần 0.3-0.5 sập nhưng chạy với lr 3e-4 nên chưa tách được lỗi do
    // offset hay do lr.
    //
    // 0.5-0.6 -> 0.0-0.4 (21-09-2026, theo yêu cầu, hướng A). Tâm cặp người cách
    // đường start->goal |offset| ~ U(0.0, 0.4) m, phía trái/phải ngẫu nhiên.
    // Lý do: (1) muốn tâm hai người sát đường đi của robot; (2) run 235537
    // (335k-412k) THỰC RA đã train ở offset 0 cố định vì lỗi offset dính (xem
    // SelectScenario) và cho 412852 PASS 4/4 ở offset=0, nên dải này bao trùm
    // điều nó đã học mà vẫn có đa dạng hình học. Bảng đo trên: 0.0 m chỉ 6% tập
    // có đường dốc học né, 0.4 m 67% - dải trộn giữ được cả hai đầu. Giữ lr
    // 1e-4 (lần 0.3-0.5 cũ sập nhưng chạy lr 3e-4). Các ô eval "offset train"
    // trước 21-09 bị nhiễm offset dính.
    //
    // 18-09-2026: dải này giờ đọc từ talking_offset_min_/max_ (mặc định lúc
    // đó 0.7/0.9, KHÔNG đổi hành vi train) thay vì hardcode, để lệnh scenario
    // "talking route ... offset lo hi" có thể ép một dải khác CHỈ cho một
    // lần gọi SelectScenario - dùng để eval "người đứng hẳn giữa đường"
    // (offset 0 0), không phải để đổi số train. Xem EnvConfig.talking_offset_override.
    const double side = (NextUniform(0.0, 1.0) < 0.5) ? 1.0 : -1.0;
    RoutePoint(NextUniform(0.45, 0.8),
      side * NextUniform(talking_offset_min_, talking_offset_max_),
      &centre_x, &centre_y);
    SpawnTalkingActor(
      "m_sweater",
      centre_x + half_gap * std::cos(axis),
      centre_y + half_gap * std::sin(axis),
      // Looking back down the axis at the partner, plus the mesh offset.
      axis + M_PI + kStandingMeshYawOffset,
      1.112927, 0.878344, 3.75, 0.0, axis + M_PI, "talking");
    SpawnTalkingActor(
      "m_mechanic",
      centre_x - half_gap * std::cos(axis),
      centre_y - half_gap * std::sin(axis),
      axis + kStandingMeshYawOffset,
      1.195040, 0.836640, 10.25, 1.25, axis, "talking");
    spawned_ = true;
    gzmsg << "Spawned talking pair at (" << centre_x << ", " << centre_y <<
      "), gap " << 2.0 * half_gap << " m\n";
  }

  // Empty the three actor lists without touching `spawned_`. The spawn
  // functions above call it; RemovePeople does the full teardown.
  void ClearScenario()
  {
    talking_actors_.clear();
    walking_actors_.clear();
    gathering_actors_.clear();
  }

  // The single entry point the trainer uses. Takes "<scenario>" or
  // "<scenario> <seed>"; seeding makes an episode reproducible when you are
  // chasing one bad rollout.
  void SelectScenario(const std::string & command)
  {
    std::istringstream stream(command);
    std::string scenario;
    stream >> scenario;

    // 21-09-2026: the `offset` token below used to STICK: talking_offset_min_/
    // max_ are members, so after one `offset 0 0` (an eval-only worst case)
    // every later command without the token -- a plain --eval, or a training
    // run on the same still-running Gazebo -- silently kept offset 0 until
    // gzserver restarted. Measured: right after an `offset 0 0` call a plain
    // `talking` command still put the pair centre 0.00 m from the route.
    // Reset to the defaults on every call; the token then applies to this
    // call only, which is what the comment on it always claimed.
    talking_offset_min_ = default_talking_offset_min_;
    talking_offset_max_ = default_talking_offset_max_;

    // Latched republish of the bare name, so a late subscriber (zone_markers.py
    // after a restart) can catch up without anyone resending the command.
    std_msgs::msg::String state_message;
    state_message.data = scenario;
    scenario_state_publisher_->publish(state_message);

    // "<scenario>", "<scenario> <seed>", "<scenario> route x0 y0 x1 y1", or
    // both. The `route` keyword rather than four bare numbers because a seed
    // is also a bare number and the two forms would be ambiguous.
    //
    // WHY THE ROUTE IS SENT AT ALL (31-08-2026). The default route below is
    // fixed, but ros_env draws a fresh start pose and goal every episode from
    // 4 x 4 = 16 combinations. Placing people around the fixed line while the
    // robot drives a different one means the carefully chosen lateral offset
    // in SpawnTalkingPeople is an offset from the WRONG line.
    //
    // Measured on the 16 combinations: with the pair offset 0.4-1.4 m from the
    // nominal route, 30.6% of episodes still put it within 0.5 m of the line
    // the robot actually drives -- dead centre, where the constraint field is
    // a plateau with a cliff and there is no gradient to learn from. The
    // policy at checkpoint 197500 failed exactly 31% of `talking` episodes
    // (5/16, all of them driving straight through at 0.20 m from a person
    // while the other 11 detoured cleanly at 1.70 m). The numbers match
    // because they are the same episodes.
    //
    // With the real route sent per episode that bucket goes to ~0.
    std::string token;
    while (stream >> token) {
      if (token == "route") {
        double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
        if (stream >> x0 >> y0 >> x1 >> y1) {
          route_start_x_ = x0;
          route_start_y_ = y0;
          route_end_x_ = x1;
          route_end_y_ = y1;
        } else {
          gzerr << "scenario command [" << command << "] has `route` without "
                << "four numbers after it; keeping the previous route\n";
        }
        continue;
      }
      // 18-09-2026: EVAL-ONLY override of the `talking` pair's lateral
      // offset band, e.g. "offset 0.0 0.0" to plant the pair dead-centre on
      // the route -- the worst case for probing avoidance, not something
      // any training run sends. See EnvConfig.talking_offset_override on the
      // Python side; default here (0.0, 0.4) is untouched unless this token
      // arrives.
      if (token == "offset") {
        double lo = 0.0, hi = 0.0;
        if (stream >> lo >> hi) {
          talking_offset_min_ = lo;
          talking_offset_max_ = hi;
        } else {
          gzerr << "scenario command [" << command << "] has `offset` "
                << "without two numbers after it; keeping the previous "
                << "offset band\n";
        }
        continue;
      }
      try {
        rng_.seed(static_cast<unsigned int>(std::stoul(token)));
      } catch (const std::exception &) {
        gzerr << "scenario command [" << command << "] carries [" << token
              << "], which is neither a seed, `route`, nor `offset`\n";
      }
    }

    // Always tear the previous scene down first: every spawn function refuses
    // to run while `spawned_` is set, so without this a reset would silently
    // keep the people from the episode before.
    RemovePeople();

    if (scenario.empty() || scenario == "none") {
      return;
    }
    if (scenario == "talking") {
      SpawnTalkingPeople();
    } else if (scenario == "passing") {
      // The RL mix now asks for one merged "somebody walks past" situation.
      // Half the episodes get the crossing geometry, half the head-on one;
      // both publish scene_type "passing".
      if (NextUniform(0.0, 1.0) < 0.5) {
        SpawnCrossingPeople();
      } else {
        SpawnApproachingPeople();
      }
    } else if (scenario == "crossing") {
      SpawnCrossingPeople();
    } else if (scenario == "approaching") {
      SpawnApproachingPeople();
    } else if (scenario == "backs_turned") {
      SpawnBacksTurnedPeople();
    } else if (scenario == "waiting") {
      SpawnWaitingPeople();
    } else if (scenario == "gathering") {
      SpawnGatheringPeople();
    } else if (scenario == "static_pair") {
      // The original fixed pair, kept for social_navigation's scenarios.
      SpawnPeople();
    } else {
      gzerr << "Unknown scenario [" << scenario << "]. Known: talking, "
        "passing, crossing, approaching, backs_turned, waiting, gathering, "
        "static_pair, none\n";
    }
  }

  void RemovePeople()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (!spawned_) {
      return;
    }
    for (auto & talking : talking_actors_) {
      if (talking.actor) {
        talking.actor->ResetCustomTrajectory();
        talking.actor->SetWorldPose(
          ignition::math::Pose3d(0.0, 0.0, -20.0, 0.0, 0.0, 0.0), false, false);
        talking.actor->Play();
      }
      MoveProxy(talking.name, 0.0, 0.0, kProxyHidden);
    }
    for (auto & walking : walking_actors_) {
      if (walking.actor) {
        walking.actor->ResetCustomTrajectory();
        walking.actor->SetWorldPose(
          ignition::math::Pose3d(0.0, 0.0, -20.0, 0.0, 0.0, 0.0), false, false);
        walking.actor->Play();
      }
      MoveProxy(walking.name, 0.0, 0.0, kProxyHidden);
    }
    for (auto & person : gathering_actors_) {
      if (person.actor) {
        person.actor->ResetCustomTrajectory();
        person.actor->SetWorldPose(
          ignition::math::Pose3d(0.0, 0.0, -20.0, 0.0, 0.0, 0.0), false, false);
        person.actor->Play();
      }
      MoveProxy(person.name, 0.0, 0.0, kProxyHidden);
    }
    ClearScenario();
    pending_people_.clear();
    previous_positions_.clear();
    spawned_ = false;
    gzmsg << "Removed animated conversation pair\n";
  }

  gazebo::physics::WorldPtr world_;
  gazebo::event::ConnectionPtr update_connection_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr release_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr gather_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr hide_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr scenario_subscription_;
  rclcpp::Publisher<social_perception::msg::People>::SharedPtr people_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr scenario_state_publisher_;
  std::vector<TalkingActor> talking_actors_;
  std::vector<WalkingActor> walking_actors_;
  std::vector<GatheringActor> gathering_actors_;
  std::vector<PersonState> pending_people_;
  std::unordered_map<std::string, std::pair<double, double>> previous_positions_;
  std::set<std::string> missing_proxies_;
  std::mutex actor_mutex_;
  std::mt19937 rng_{std::random_device{}()};
  bool spawned_{false};
  double actor_ground_z_{0.20};
  double approach_duration_{8.0};
  double talk_duration_{60.0};
  double disperse_duration_{8.0};
  double away_duration_{20.0};
  double pair_separation_{2.8};
  double publish_period_{0.05};
  double last_publish_time_{-1.0};
  // The nominal route the RL scenarios are laid out around, in world
  // coordinates: from the robot spawn in gazebo.launch.py to the middle of the
  // goal cluster in rl_train.yaml. Overridable from the world file so that
  // moving the goals does not leave every scenario placing people behind the
  // robot.
  double route_start_x_{-3.0};
  double route_start_y_{-2.0};
  double route_end_x_{1.2};
  double route_end_y_{0.3};

  // 18-09-2026: `talking` pair lateral offset band, in metres either side of
  // the route. Matches the trained default (0.0-0.4); only a `offset lo hi`
  // token in SelectScenario's command changes it, and that token is only
  // ever sent for an eval-only worst case, never during training. See
  // EnvConfig.talking_offset_override.
  const double default_talking_offset_min_{0.0};
  const double default_talking_offset_max_{0.4};
  double talking_offset_min_{0.0};
  double talking_offset_max_{0.4};
};

GZ_REGISTER_WORLD_PLUGIN(AnimatedPeopleRelease)
}  // namespace linorobot2_gazebo
