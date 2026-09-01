#ifndef SOCIAL_NAVIGATION__SOCIAL_LAYER_HPP_
#define SOCIAL_NAVIGATION__SOCIAL_LAYER_HPP_

#include <mutex>
#include <string>

#include "nav2_costmap_2d/costmap_layer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/lifecycle_publisher.hpp"
#include "social_perception/msg/people.hpp"

namespace social_navigation
{

class SocialLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  SocialLayer() = default;
  void onInitialize() override;
  void updateBounds(double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}

private:
  void peopleCallback(const social_perception::msg::People::SharedPtr message);
  rclcpp::Subscription<social_perception::msg::People>::SharedPtr people_sub_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::OccupancyGrid>::SharedPtr social_grid_pub_;
  social_perception::msg::People people_;
  rclcpp::Time people_received_at_{0, 0, RCL_ROS_TIME};
  bool have_people_message_{false};
  std::mutex data_mutex_;
  double cutoff_{10.0};
  double amplitude_{255.0};
  double covariance_front_height_{0.4};
  double covariance_front_width_{0.25};
  double covariance_rear_height_{0.25};
  double covariance_rear_width_{0.25};
  double covariance_right_height_{0.3};
  double covariance_right_width_{0.2};
  double covariance_when_still_{0.25};
  bool use_passing_{true};
  bool use_vel_factor_{true};
  double speed_factor_multiplier_{5.0};
  double velocity_still_threshold_{0.1};
  bool publish_occgrid_{false};
  double data_timeout_{1.0};
  double last_min_x_{0.0};
  double last_min_y_{0.0};
  double last_max_x_{0.0};
  double last_max_y_{0.0};
  bool have_last_bounds_{false};
};

}  // namespace social_navigation
#endif
