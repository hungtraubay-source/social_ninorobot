#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gazebo/common/Console.hh>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Actor.hh>
#include <gazebo/physics/World.hh>
#include <gazebo_ros/node.hpp>
#include <social_perception/msg/people.hpp>
#include <social_perception/msg/person.hpp>
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

    // How far apart the two people stand while talking, centre to centre.
    //
    // This decides whether the robot can physically drive between them, which
    // is what makes the social region testable: with a gap it cannot fit
    // through, a run proves nothing about the region, because geometry alone
    // already turned the robot away.
    //
    // The capture default is a natural close conversation: 1.5 m from centre
    // to centre. A world file may still override it with <pair_separation>.
    pair_separation_ = sdf->Get<double>("pair_separation", 1.5).first;

    update_connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(
      [this](const gazebo::common::UpdateInfo &) { UpdateActors(); });
    release_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/release", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { SpawnPeople(); });
    gather_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/gather", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { SpawnGatheringPeople(); });
    crossing_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/crossing", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { SpawnCrossingPerson(); });
    // A stationary single-person scenario is useful for checking depth and
    // social-space stability without walking velocity or route resets.
    standing_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/standing", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { SpawnStandingPerson(); });
    hide_subscription_ = node_->create_subscription<std_msgs::msg::Empty>(
      "/animated_people/hide", rclcpp::QoS(10),
      [this](std_msgs::msg::Empty::ConstSharedPtr) { RemovePeople(); });
    ground_truth_publisher_ = node_->create_publisher<social_perception::msg::People>(
      "/scenario/ground_truth", rclcpp::QoS(10));
    scenario_state_publisher_ = node_->create_publisher<std_msgs::msg::String>(
      "/scenario/state", rclcpp::QoS(10));
    track_reset_publisher_ = node_->create_publisher<std_msgs::msg::Empty>(
      "/animated_people/track_reset", rclcpp::QoS(10));
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
    double start_time{0.0};
    double last_route_time{0.0};
    bool route_time_initialized{false};
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
    // ``true`` keeps one skeletal frame for the standing test.  The actor
    // remains visually human-shaped, but its arms and torso cannot perturb
    // RGB-D bounding-box/depth localization from one camera frame to another.
    bool freeze_animation{false};
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

  void SpawnTalkingActor(
    const std::string & name,
    double x, double y, double yaw,
    double root_roll, double root_height,
    double animation_duration, double phase_offset,
    bool freeze_animation = false)
  {
    talking_actors_.push_back({
      name, x, y, 0.20 + root_height, yaw, root_roll,
      animation_duration, phase_offset, freeze_animation});
  }

  void SpawnWalkingActor(
    const std::string & actor_name,
    const std::vector<WalkingWaypoint> & waypoints,
    double z = 0.20)
  {
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
    walking_actors_.push_back({actor_name, waypoints, z, 1.416667, 1.206206});
  }

  static double InterpolateYaw(double from, double to, double ratio)
  {
    const double difference = std::atan2(std::sin(to - from), std::cos(to - from));
    return from + ratio * difference;
  }

  void UpdateActors()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (!spawned_) {
      return;
    }
    const double now = world_->SimTime().Double();

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
        talking.start_time = now;
        talking.actor->Play();
        gzmsg << "Talking actor [" << talking.name << "] started\n";
      }

      // Reapply the same script time for a static actor.  Gazebo's actor
      // skeleton otherwise advances the ``talk`` clip even though its world
      // pose is fixed, moving hands and corrupting the RGB-D torso depth.
      const double elapsed = std::max(0.0, now - talking.start_time);
      const double animation_time = talking.freeze_animation ?
        talking.phase_offset : std::fmod(
          elapsed + talking.phase_offset, talking.animation_duration);
      talking.actor->SetScriptTime(animation_time);
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
      const double route_time = std::fmod(std::max(0.0, now - walking.start_time), duration);
      // The crossing route deliberately teleports from B back to A. Notify
      // perception so it drops the old endpoint instead of coasting it as a
      // second person while YOLO detects the restarted actor at A.
      if (walking.route_time_initialized && route_time < walking.last_route_time) {
        track_reset_publisher_->publish(std_msgs::msg::Empty());
        gzmsg << "Walking actor [" << walking.name << "] restarted route\n";
      }
      walking.last_route_time = route_time;
      walking.route_time_initialized = true;
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
          x, y, walking.z + 0.886948, walking.root_roll, 0.0, yaw), false, false);
    }

    UpdateGatheringActors(now);
    PublishGroundTruth();
  }

  void PublishGroundTruth()
  {
    social_perception::msg::People ground_truth;
    // Actor poses come from Gazebo's world frame.  Keep that frame explicit:
    // consumers can transform it through the existing world -> map TF.
    ground_truth.header.stamp = node_->get_clock()->now();
    ground_truth.header.frame_id = "world";

    auto add_actor = [&ground_truth](const std::string & id,
                                     const gazebo::physics::ActorPtr & actor) {
      if (!actor) {
        return;
      }
      const auto pose = actor->WorldPose();
      // The gathering scenario parks its actors at z=-20 while they are out
      // of scene.  They are not valid labels for a visible frame.
      if (pose.Pos().Z() < -10.0) {
        return;
      }
      social_perception::msg::Person person;
      person.id = id;
      person.pose.position.x = pose.Pos().X();
      person.pose.position.y = pose.Pos().Y();
      person.pose.position.z = pose.Pos().Z();
      person.pose.orientation.x = pose.Rot().X();
      person.pose.orientation.y = pose.Rot().Y();
      person.pose.orientation.z = pose.Rot().Z();
      person.pose.orientation.w = pose.Rot().W();
      ground_truth.people.push_back(person);
    };
    for (const auto & actor : talking_actors_) {
      add_actor(actor.name, actor.actor);
    }
    for (const auto & actor : walking_actors_) {
      add_actor(actor.name, actor.actor);
    }
    for (const auto & actor : gathering_actors_) {
      add_actor(actor.name, actor.actor);
    }
    ground_truth_publisher_->publish(ground_truth);
    std_msgs::msg::String state;
    state.data = scenario_state_;
    scenario_state_publisher_->publish(state);
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

      scenario_state_ = (phase == 0) ? "approaching" :
        (phase == 1) ? "talking" : (phase == 2) ? "dispersing" : "away";

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
          x, y, 0.20 + animation.root_height,
          animation.root_roll, 0.0, yaw),
        false, false);
    }
  }

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
    SpawnTalkingActor(
      "m_sweater", 0.0, half_gap, 0.0, 1.112927, 0.878344, 3.75, 0.0);
    SpawnTalkingActor(
      "m_mechanic", 0.0, -half_gap, 3.14159265359,
      1.195040, 0.836640, 10.25, 1.25);

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
    scenario_state_ = "talking";
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

    scenario_state_ = "approaching";
    spawned_ = true;
    gzmsg << "Spawned gathering scenario: approach " << approach_duration_ <<
      "s, talk " << talk_duration_ << "s, disperse " << disperse_duration_ <<
      "s, away " << away_duration_ << "s\n";
  }

  void SpawnCrossingPerson()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    talking_actors_.clear();
    walking_actors_.clear();
    gathering_actors_.clear();

    // The actor walks from A (-y) to B (+y) at world x=+1.  With the default
    // robot spawn at world x=-3, this is approximately map x=+4: open floor
    // beyond the narrow doorway instead of its centre.  This keeps the person
    // visible to the camera while allowing Nav2 a real lateral detour to test.
    // The walking-route timer wraps at the final waypoint, so it is teleported
    // to A at 12 s and immediately starts another A -> B pass; it never walks
    // back from B to A.
    SpawnWalkingActor(
      "walker_1",
      {
        {0.0, 1.0, -2.0, 3.14159265359},
        {12.0, 1.0, 2.0, 3.14159265359},
      });
    scenario_state_ = "crossing";
    spawned_ = true;
    gzmsg << "Spawned crossing scenario: walker_1 crosses the camera every 12 s\n";
  }

  void SpawnStandingPerson()
  {
    std::lock_guard<std::mutex> lock(actor_mutex_);
    if (spawned_) {
      return;
    }
    talking_actors_.clear();
    walking_actors_.clear();
    gathering_actors_.clear();

    // Coordinates are Gazebo world metres, matching the existing talking and
    // crossing scenarios. ``SpawnTalkingActor`` applies m_sweater's skeletal
    // root offset so its feet rest on the cafe floor instead of using z=0 for
    // the actor root (which would bury the animated mesh below the surface).
    // yaw=pi follows this actor model's facing convention and points it toward
    // the bookshelf at world (0, 0), while
    // it remains at (0, -1.4) on the XY plane for the terminal session.
    // Freeze the talk clip at time zero: the model has no native idle clip,
    // and a frozen skeleton prevents arm gestures from destabilising depth.
    SpawnTalkingActor(
      "m_sweater", 0.0, -1.4, 3.14, 1.112927, 0.878344, 3.75, 0.0, true);

    scenario_state_ = "standing";
    spawned_ = true;
    gzmsg << "Spawned standing scenario: m_sweater at world (0, -1.4, floor)\n";
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
    }
    for (auto & walking : walking_actors_) {
      if (walking.actor) {
        walking.actor->ResetCustomTrajectory();
        walking.actor->SetWorldPose(
          ignition::math::Pose3d(0.0, 0.0, -20.0, 0.0, 0.0, 0.0), false, false);
        walking.actor->Play();
      }
    }
    for (auto & person : gathering_actors_) {
      if (person.actor) {
        person.actor->ResetCustomTrajectory();
        person.actor->SetWorldPose(
          ignition::math::Pose3d(0.0, 0.0, -20.0, 0.0, 0.0, 0.0), false, false);
        person.actor->Play();
      }
    }
    talking_actors_.clear();
    walking_actors_.clear();
    gathering_actors_.clear();
    scenario_state_ = "none";
    spawned_ = false;
    gzmsg << "Removed animated people\n";
  }

  gazebo::physics::WorldPtr world_;
  gazebo::event::ConnectionPtr update_connection_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr release_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr gather_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr crossing_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr standing_subscription_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr hide_subscription_;
  std::vector<TalkingActor> talking_actors_;
  std::vector<WalkingActor> walking_actors_;
  std::vector<GatheringActor> gathering_actors_;
  rclcpp::Publisher<social_perception::msg::People>::SharedPtr ground_truth_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr scenario_state_publisher_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr track_reset_publisher_;
  std::mutex actor_mutex_;
  bool spawned_{false};
  double approach_duration_{8.0};
  double talk_duration_{60.0};
  double disperse_duration_{8.0};
  double away_duration_{20.0};
  // Centre-to-centre distance of the two actors while conversing.
  double pair_separation_{1.5};
  std::string scenario_state_{"none"};
};

GZ_REGISTER_WORLD_PLUGIN(AnimatedPeopleRelease)
}  // namespace linorobot2_gazebo
