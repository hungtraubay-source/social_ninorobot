import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime
import numpy as np

class RealRobotFBLFollower(Node):
    def __init__(self):
        # 1. DÙNG THỜI GIAN THỰC (Real Time)
        super().__init__(
            'fbl_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)
            ]
        )

        # 2. PARAMETERS 
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('Kp', 1.2)
        self.declare_parameter('Kd', 1.2)
        self.declare_parameter('num_loops', 10)

        self.off_x = self.get_parameter('offset_x').value
        self.off_y = self.get_parameter('offset_y').value
        self.off_phi = self.get_parameter('offset_phi').value
        self.Ts = self.get_parameter('Ts').value
        self.K = np.array([self.get_parameter('Kp').value, self.get_parameter('Kd').value])
        self.num_loops = self.get_parameter('num_loops').value

        self.get_logger().info(f"REAL ROBOT: Ts={self.Ts}s. Offset: {self.off_x}, {self.off_y}, {self.off_phi}")

        # 3. TRAJECTORY CONSTANTS
        self.A = 0.7
        self.s_end_one = 30.0
        self.s_end = self.s_end_one * self.num_loops
        self.freq = 2 * math.pi / self.s_end_one
        self.v_path_max = 0.8
        self.a_path_max = 0.5

        # 4. STATE & PHYSICAL LIMITS
        self.s = 0.0
        self.v_s = 0.01
        self.curr_q = [self.off_x, self.off_y, self.off_phi]
        self.v_robot = 0.01   # Vận tốc hồi tiếp từ Odom
        self.v_prev, self.w_prev = 0.0, 0.0
        
        self.v_max, self.w_max = 0.3, 3.85
        self.dv_max, self.dw_max = 1.0, 4.0

        # REAL ROBOT
        self.K = [0.2, 0.2]
        self.v_max = 0.2
        self.w_max = 1.5
        self.dv_max = 0.25
        self.dw_max = 1.0
        self.v_path_max = 0.5
        self.a_path_max = 0.3

        self.start_time = None
        self.odom_received = False
        self.sum_e_dist_sq = 0.0
        self.count = 0
        self.last_log_t = -1
        self.data_log = []
        self.csv_filename = f'real_motion_law_fbl_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # 5. ROS INFRASTRUCTURE
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        qos_p = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_p)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'

        # 6. TIMERS
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.timer = self.create_timer(self.Ts, self.control_loop)
        
        self.get_logger().info(f"FBL Follower Ready.")

    def publish_static_ref_path(self):
        msg = Path()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(400):
            ss = (i / 400.0) * self.s_end_one
            p = PoseStamped()
            p.header.frame_id = 'world'
            p.pose.position.x = 1.1 + self.A * math.sin(self.freq * ss)
            p.pose.position.y = 0.9 + self.A * math.sin(2 * self.freq * ss)
            msg.poses.append(p)
        self.ref_path_pub.publish(msg)

    def odom_callback(self, msg):
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ophi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        c, s = math.cos(self.off_phi), math.sin(self.off_phi)
        self.curr_q[0] = ox * c - oy * s + self.off_x
        self.curr_q[1] = ox * s + oy * c + self.off_y
        self.curr_q[2] = math.atan2(math.sin(ophi + self.off_phi), math.cos(ophi + self.off_phi))
        
        # Vận tốc robot (Dùng bộ lọc nhiễu nhẹ nếu cần)
        self.v_robot = msg.twist.twist.linear.x
        self.odom_received = True

    def control_loop(self):
        if not self.odom_received: return
        now = self.get_clock().now()
        if self.start_time is None: self.start_time = now; return
        t = (now - self.start_time).nanoseconds / 1e9

        # A. MOTION LAW s(t)
        dist_to_end = self.s_end - self.s
        if self.s < 1.0: a_s = self.a_path_max
        elif dist_to_end < 0.5: a_s = -self.a_path_max
        else: a_s = 0.0
        self.v_s = max(min(self.v_s + a_s * self.Ts, self.v_path_max), 0.02)
        self.s += self.v_s * self.Ts
        if self.s >= self.s_end: 
            self.stop_robot()
            return

        # B. REFERENCE & CHAIN RULE
        ss, vs, as_v = self.s, self.v_s, a_s
        f = self.freq
        x_ref = 1.1 + self.A * math.sin(f * ss)
        y_ref = 0.9 + self.A * math.sin(2 * f * ss)
        dx_ds = self.A * f * math.cos(f * ss)
        dy_ds = 2 * self.A * f * math.cos(2 * f * ss)
        dx_ref, dy_ref = dx_ds * vs, dy_ds * vs
        d2x_ds2 = -self.A * (f**2) * math.sin(f * ss)
        d2y_ds2 = -4 * self.A * (f**2) * math.sin(2 * f * ss)
        ddx_ref, ddy_ref = d2x_ds2*(vs**2) + dx_ds*as_v, d2y_ds2*(vs**2) + dy_ds*as_v

        # C. FBL LOGIC
        phi = self.curr_q[2]
        # Trên robot thật, dùng v_prev để tính ma trận F sẽ ổn định hơn dùng v_robot từ odom nhiễu
        v_feedback = self.v_prev 
        ez1 = np.array([x_ref - self.curr_q[0], dx_ref - v_feedback * math.cos(phi)])
        ez2 = np.array([y_ref - self.curr_q[1], dy_ref - v_feedback * math.sin(phi)])
        
        ux = ddx_ref + np.dot(self.K, ez1)
        uy = ddy_ref + np.dot(self.K, ez2)
        
        # Singularity handling
        v_safe = max(abs(v_feedback), 0.08) * np.sign(v_feedback if v_feedback != 0 else 1)
        
        v_dot = ux * math.cos(phi) + uy * math.sin(phi)
        w_target = (-ux * math.sin(phi) + uy * math.cos(phi)) / v_safe

        # D. RATE LIMIT
        v_accel = max(min(v_dot, self.dv_max), -self.dv_max)
        v_final = max(min(self.v_prev + v_accel * self.Ts, self.v_max), -self.v_max)
        v_final = max(min(v_final, self.v_max), -self.v_max)
        
        w_target = max(min(w_target, self.w_max), -self.w_max)
        delta_w = max(min(w_target - self.w_prev, self.dw_max * self.Ts), -self.dw_max * self.Ts)
        w_final = self.w_prev + delta_w

        # E. EXECUTE
        self.v_prev, self.w_prev = v_final, w_final
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_final, w_final
        self.cmd_pub.publish(cmd)
        
        e_dist = math.sqrt((x_ref - self.curr_q[0])**2 + (y_ref - self.curr_q[1])**2)
        self.update_path_and_log(t, e_dist, v_final, w_final)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info("=== STOPPED ===")

    def update_path_and_log(self, t, e_dist, v, w):
        p = PoseStamped()
        p.header.frame_id, p.header.stamp = 'world', self.get_clock().now().to_msg()
        p.pose.position.x, p.pose.position.y = self.curr_q[0], self.curr_q[1]
        self.actual_path.poses.append(p)
        if len(self.actual_path.poses) > 2000: self.actual_path.poses.pop(0)
        self.actual_path_pub.publish(self.actual_path)

        self.sum_e_dist_sq += (e_dist ** 2)
        self.count += 1
        rms = math.sqrt(self.sum_e_dist_sq / self.count)
        self.data_log.append([t, e_dist, rms, v, w])
        if int(t) > self.last_log_t:
            self.get_logger().info(f"Time: {t:.1f}s | Err: {e_dist:.3f}m | RMS: {rms:.3f}m | v: {v:.3f}m/s | w: {w:.3f}rad/s")
            self.last_log_t = int(t)

    def save_csv(self):
        if self.data_log:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'Error (m)', 'RMS_Err (m)', 'v (m/s)', 'w (rad/s)'])
                writer.writerows(self.data_log)
            self.get_logger().info(f"Log: {self.csv_filename}")

def main():
    rclpy.init()
    node = RealRobotFBLFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop_robot()
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()