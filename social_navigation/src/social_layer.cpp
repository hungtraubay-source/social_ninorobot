#include "social_navigation/social_layer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace social_navigation
{

namespace
{

constexpr double kHalfPi = 1.57079632679489661923;

double gaussian(
  double x, double y, double center_x, double center_y, double amplitude,
  double covariance_heading, double covariance_lateral, double heading)
{
  const double dx = x - center_x;
  const double dy = y - center_y;
  const double forward = std::cos(heading) * dx + std::sin(heading) * dy;
  const double lateral = -std::sin(heading) * dx + std::cos(heading) * dy;
  return amplitude * std::exp(-(
           (forward * forward) / (2.0 * covariance_heading) +
           (lateral * lateral) / (2.0 * covariance_lateral)));
}

double gaussianRadius(double cutoff, double amplitude, double covariance)
{
  return std::sqrt(-2.0 * covariance * std::log(cutoff / amplitude));
}

double angularDistance(double from, double to)
{
  return std::atan2(std::sin(to - from), std::cos(to - from));
}

}  // namespace

void SocialLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("SocialLayer cannot lock lifecycle node");
  }
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("people_topic", rclcpp::ParameterValue(std::string("/people")));
  declareParameter("cutoff", rclcpp::ParameterValue(10.0));
  declareParameter("amplitude", rclcpp::ParameterValue(255.0));
  declareParameter("covariance_front_height", rclcpp::ParameterValue(0.4));
  declareParameter("covariance_front_width", rclcpp::ParameterValue(0.25));
  declareParameter("covariance_rear_height", rclcpp::ParameterValue(0.25));
  declareParameter("covariance_rear_width", rclcpp::ParameterValue(0.25));
  declareParameter("covariance_right_height", rclcpp::ParameterValue(0.3));
  declareParameter("covariance_right_width", rclcpp::ParameterValue(0.2));
  declareParameter("covariance_when_still", rclcpp::ParameterValue(0.25));
  declareParameter("use_passing", rclcpp::ParameterValue(true));
  declareParameter("use_vel_factor", rclcpp::ParameterValue(true));
  declareParameter("speed_factor_multiplier", rclcpp::ParameterValue(5.0));
  declareParameter("velocity_still_threshold", rclcpp::ParameterValue(0.1));
  declareParameter("publish_occgrid", rclcpp::ParameterValue(false));
  declareParameter("data_timeout", rclcpp::ParameterValue(1.0));

  std::string people_topic;
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".people_topic", people_topic);
  node->get_parameter(name_ + ".cutoff", cutoff_);
  node->get_parameter(name_ + ".amplitude", amplitude_);
  node->get_parameter(name_ + ".covariance_front_height", covariance_front_height_);
  node->get_parameter(name_ + ".covariance_front_width", covariance_front_width_);
  node->get_parameter(name_ + ".covariance_rear_height", covariance_rear_height_);
  node->get_parameter(name_ + ".covariance_rear_width", covariance_rear_width_);
  node->get_parameter(name_ + ".covariance_right_height", covariance_right_height_);
  node->get_parameter(name_ + ".covariance_right_width", covariance_right_width_);
  node->get_parameter(name_ + ".covariance_when_still", covariance_when_still_);
  node->get_parameter(name_ + ".use_passing", use_passing_);
  node->get_parameter(name_ + ".use_vel_factor", use_vel_factor_);
  node->get_parameter(name_ + ".speed_factor_multiplier", speed_factor_multiplier_);
  node->get_parameter(name_ + ".velocity_still_threshold", velocity_still_threshold_);
  node->get_parameter(name_ + ".publish_occgrid", publish_occgrid_);
  node->get_parameter(name_ + ".data_timeout", data_timeout_);

  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;
  people_sub_ = node->create_subscription<social_perception::msg::People>(
    people_topic, rclcpp::QoS(10),
    std::bind(&SocialLayer::peopleCallback, this, std::placeholders::_1), options);
  if (publish_occgrid_) {
    social_grid_pub_ = node->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "social_grid", rclcpp::QoS(1).transient_local());
    social_grid_pub_->on_activate();
  }
  current_ = true;
  matchSize();
  RCLCPP_INFO(logger_, "SocialLayer listening on %s", people_topic.c_str());
}

void SocialLayer::peopleCallback(
  const social_perception::msg::People::SharedPtr message)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  people_ = *message;
  people_received_at_ = clock_->now();
  have_people_message_ = true;
}

void SocialLayer::reset()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  people_.people.clear();
  have_people_message_ = false;
  have_last_bounds_ = false;
  current_ = true;
}

void SocialLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {return;}
  social_perception::msg::People people;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    people = people_;
    const auto now = clock_->now();
    if (!have_people_message_ || (now - people_received_at_).seconds() > data_timeout_) {
      people.people.clear();
    }
  }

  if (have_last_bounds_) {
    *min_x = std::min(*min_x, last_min_x_);
    *min_y = std::min(*min_y, last_min_y_);
    *max_x = std::max(*max_x, last_max_x_);
    *max_y = std::max(*max_y, last_max_y_);
  }

  double new_min_x = std::numeric_limits<double>::max();
  double new_min_y = std::numeric_limits<double>::max();
  double new_max_x = std::numeric_limits<double>::lowest();
  double new_max_y = std::numeric_limits<double>::lowest();
  bool found = false;
  const std::string target = layered_costmap_->getGlobalFrameID();

  auto include_point = [&](double x, double y, double radius) {
      new_min_x = std::min(new_min_x, x - radius);
      new_min_y = std::min(new_min_y, y - radius);
      new_max_x = std::max(new_max_x, x + radius);
      new_max_y = std::max(new_max_y, y + radius);
      found = true;
    };

  try {
    for (const auto & person : people.people) {
      geometry_msgs::msg::PoseStamped input;
      input.header = people.header;
      input.pose = person.pose;
      const auto output = tf_->transform(input, target, tf2::durationFromSec(0.1));
      geometry_msgs::msg::Vector3Stamped velocity_input;
      velocity_input.header = people.header;
      velocity_input.vector = person.velocity.linear;
      const auto velocity = tf_->transform(
        velocity_input, target, tf2::durationFromSec(0.1));
      const double speed = std::hypot(velocity.vector.x, velocity.vector.y);
      double greatest_covariance = covariance_when_still_;
      if (speed >= velocity_still_threshold_) {
        const double speed_factor = use_vel_factor_ ?
          1.0 + speed * speed_factor_multiplier_ : 1.0;
        greatest_covariance = std::max({
            covariance_front_height_ * speed_factor,
            covariance_front_width_, covariance_rear_height_, covariance_rear_width_,
            use_passing_ ? covariance_right_height_ : 0.0,
            use_passing_ ? covariance_right_width_ : 0.0});
      }
      include_point(output.pose.position.x, output.pose.position.y,
        gaussianRadius(cutoff_, amplitude_, greatest_covariance));
    }
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "SocialLayer transform failed: %s", error.what());
  }

  if (found) {
    last_min_x_ = new_min_x;
    last_min_y_ = new_min_y;
    last_max_x_ = new_max_x;
    last_max_y_ = new_max_y;
    have_last_bounds_ = true;
    *min_x = std::min(*min_x, new_min_x);
    *min_y = std::min(*min_y, new_min_y);
    *max_x = std::max(*max_x, new_max_x);
    *max_y = std::max(*max_y, new_max_y);
  } else {
    have_last_bounds_ = false;
  }
}

void SocialLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {return;}
  social_perception::msg::People people;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    people = people_;
    const auto now = clock_->now();
    if (!have_people_message_ || (now - people_received_at_).seconds() > data_timeout_) {
      people.people.clear();
    }
  }

  struct PersonInMap {double x; double y; double vx; double vy; double speed;};
  std::vector<PersonInMap> mapped_people;
  const std::string target = layered_costmap_->getGlobalFrameID();
  try {
    for (const auto & person : people.people) {
      geometry_msgs::msg::PoseStamped input;
      input.header = people.header;
      input.pose = person.pose;
      const auto output = tf_->transform(input, target, tf2::durationFromSec(0.1));
      geometry_msgs::msg::Vector3Stamped velocity_input;
      velocity_input.header = people.header;
      velocity_input.vector = person.velocity.linear;
      const auto velocity = tf_->transform(
        velocity_input, target, tf2::durationFromSec(0.1));
      mapped_people.push_back({output.pose.position.x, output.pose.position.y,
        velocity.vector.x, velocity.vector.y,
        std::hypot(velocity.vector.x, velocity.vector.y)});
    }
  } catch (const tf2::TransformException & error) {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 2000, "SocialLayer transform failed: %s", error.what());
    return;
  }

  const int begin_i = std::max(0, min_i);
  const int begin_j = std::max(0, min_j);
  const int end_i = std::min(static_cast<int>(master.getSizeInCellsX()), max_i);
  const int end_j = std::min(static_cast<int>(master.getSizeInCellsY()), max_j);
  resetMap(begin_i, begin_j, end_i, end_j);

  nav_msgs::msg::OccupancyGrid social_grid;
  if (social_grid_pub_) {
    social_grid.header.stamp = clock_->now();
    social_grid.header.frame_id = target;
    social_grid.info.resolution = master.getResolution();
    social_grid.info.width = master.getSizeInCellsX();
    social_grid.info.height = master.getSizeInCellsY();
    social_grid.info.origin.position.x = master.getOriginX();
    social_grid.info.origin.position.y = master.getOriginY();
    social_grid.info.origin.orientation.w = 1.0;
    social_grid.data.assign(
      static_cast<size_t>(social_grid.info.width) * social_grid.info.height, -1);
  }
  for (int j = begin_j; j < end_j; ++j) {
    for (int i = begin_i; i < end_i; ++i) {
      double wx, wy;
      master.mapToWorld(i, j, wx, wy);
      unsigned char social_cost = nav2_costmap_2d::FREE_SPACE;

      for (const auto & person : mapped_people) {
        double cost = 0.0;
        if (person.speed < velocity_still_threshold_) {
          cost = gaussian(
            wx, wy, person.x, person.y, amplitude_, covariance_when_still_,
            covariance_when_still_, 0.0);
        } else {
          const double motion_yaw = std::atan2(person.vy, person.vx);
          const double point_yaw = std::atan2(wy - person.y, wx - person.x);
          const double heading_difference = angularDistance(motion_yaw, point_yaw);
          const double front_factor = use_vel_factor_ ?
            1.0 + person.speed * speed_factor_multiplier_ : 1.0;
          if (std::fabs(heading_difference) < kHalfPi) {
            cost = gaussian(
              wx, wy, person.x, person.y, amplitude_,
              covariance_front_height_ * front_factor, covariance_front_width_, motion_yaw);
          } else {
            cost = gaussian(
              wx, wy, person.x, person.y, amplitude_, covariance_rear_height_,
              covariance_rear_width_, motion_yaw);
          }
          if (use_passing_) {
            const double passing_yaw = motion_yaw - kHalfPi;
            const double passing_difference = angularDistance(passing_yaw, point_yaw);
            if (std::fabs(passing_difference) < kHalfPi) {
              cost = std::max(cost, gaussian(
                  wx, wy, person.x, person.y, amplitude_, covariance_right_height_,
                  covariance_right_width_, passing_yaw));
            }
          }
        }
        if (cost >= cutoff_) {
          social_cost = std::max(social_cost, static_cast<unsigned char>(
              std::clamp(static_cast<int>(std::round(cost)), 0, 254)));
        }
      }

      if (social_cost > nav2_costmap_2d::FREE_SPACE) {setCost(i, j, social_cost);}
      if (social_grid_pub_ && social_cost > nav2_costmap_2d::FREE_SPACE) {
        social_grid.data[master.getIndex(i, j)] = static_cast<int8_t>(std::round(
            static_cast<double>(social_cost) * 100.0 / 254.0));
      }
    }
  }
  updateWithMax(master, begin_i, begin_j, end_i, end_j);
  if (social_grid_pub_) {social_grid_pub_->publish(social_grid);}
}

}  // namespace social_navigation

PLUGINLIB_EXPORT_CLASS(social_navigation::SocialLayer, nav2_costmap_2d::Layer)
