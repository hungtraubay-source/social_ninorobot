import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime
import numpy as np

class FBLFollower(Node):
    def __init__(self):
        super().__init__(
            'fbl_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # 1. PARAMETERS
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('Kp', 1)
        self.declare_parameter('Kd', 1)
        self.declare_parameter('num_loops', 2)

        self.off_x = self.get_parameter('offset_x').value
        self.off_y = self.get_parameter('offset_y').value
        self.off_phi = self.get_parameter('offset_phi').value
        self.Ts = self.get_parameter('Ts').value
        self.K = np.array([self.get_parameter('Kd').value, self.get_parameter('Kp').value])
        self.num_loops = self.get_parameter('num_loops').value

        self.get_logger().info(f"Ts={self.Ts}s. Offset: {self.off_x}, {self.off_y}, {self.off_phi}")

        # 2. TRAJECTORY CONSTANTS
        self.A = 0.7
        self.s_end_one_loop = 30.0
        self.s_end = self.s_end_one_loop * self.num_loops
        self.freq = 2 * math.pi / self.s_end_one_loop
        self.v_path_max = 0.8
        self.a_path_max = 0.5

        # 3. STATE VARIABLES
        self.s = 0.0
        self.v_s = 0.01
        self.curr_q = [self.off_x, self.off_y, self.off_phi]
        self.v_robot = 0.01 # Vận tốc dài hiện tại
        self.v_prev, self.w_prev = 0.0, 0.0
        
        self.v_max, self.w_max = 0.33, 3.85
        self.dv_max, self.dw_max = 1.0, 4.0

        # Modified
        self.K = np.array([5.0, 4.0])
        self.v_max = 0.33
        self.w_max = 3.85
        self.dv_max = 1.0
        self.dw_max = 4.0
        self.v_path_max = 0.8
        self.a_path_max = 0.5

        self.start_time = None
        self.odom_received = False
        self.sum_e_dist_sq = 0.0
        self.count = 0
        self.last_log_t = -1
        self.data_log = []
        self.csv_filename = f'motion_law_fbl_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # 4. ROS INFRASTRUCTURE
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        qos_p = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_p)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.timer = self.create_timer(self.Ts, self.control_loop)
        
        self.get_logger().info(f"FBL Follower initialized. Play Gazebo!")

    def publish_static_ref_path(self):
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp = self.get_clock().now().to_msg()
        for i in range(400):
            ss = (i / 400.0) * self.s_end_one_loop
            p = PoseStamped()
            p.header.frame_id = 'world'
            p.pose.position.x = 1.1 + self.A * math.sin(self.freq * ss)
            p.pose.position.y = 0.9 + self.A * math.sin(2 * self.freq * ss)
            ref_path.poses.append(p)
        self.ref_path_pub.publish(ref_path)

    def odom_callback(self, msg):
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ophi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        c, s = math.cos(self.off_phi), math.sin(self.off_phi)
        self.curr_q[0] = ox * c - oy * s + self.off_x
        self.curr_q[1] = ox * s + oy * c + self.off_y
        self.curr_q[2] = math.atan2(math.sin(ophi + self.off_phi), math.cos(ophi + self.off_phi))
        
        # Lấy vận tốc hiện tại từ Odom để dùng cho FBL
        self.v_robot = msg.twist.twist.linear.x
        self.odom_received = True

    def get_lookahead_safe_velocity(self, s_curr):
        """ Quét một đoạn look-ahead phía trước để tìm vận tốc an toàn nhỏ nhất """
        look_ahead_dist = 0.3
        num_samples = 5        # Số điểm lấy mẫu
        beta = 0.3             # Hệ số nhạy cảm (Tăng beta nếu muốn cua chậm hơn)
        
        v_min_found = self.v_path_max

        for i in range(num_samples + 1):
            s_future = s_curr + (i / num_samples) * look_ahead_dist
            
            f = self.freq
            A = self.A
            dx = A * f * math.cos(f * s_future)
            dy = 2 * A * f * math.cos(2 * f * s_future)
            ddx = -A * (f**2) * math.sin(f * s_future)
            ddy = -4 * A * (f**2) * math.sin(2 * f * s_future)

            # Tính độ cong Kappa
            # kappa = |x_dot*y_ddot - y_dot*x_ddot| / (x_dot^2 + y_dot^2)^1.5
            v_ref_mag_sq = dx**2 + dy**2
            kappa = abs(dx * ddy - dy * ddx) / (v_ref_mag_sq**1.5 + 1e-6)
            
            # Vận tốc an toàn tại điểm tương lai
            v_safe = self.v_path_max / (1.0 + beta * kappa)
            
            if v_safe < v_min_found:
                v_min_found = v_safe
                
        return v_min_found

    def control_loop(self):
        if not self.odom_received: return
        now = self.get_clock().now()
        if self.start_time is None: self.start_time = now; return
        t = (now - self.start_time).nanoseconds / 1e9

         # --- LOOK-AHEAD ADAPTIVE VELOCITY ---
        v_path_adaptive = self.get_lookahead_safe_velocity(self.s)

        # A. MOTION LAW s(t)
        dist_to_end = self.s_end - self.s
        
        if dist_to_end < 0.5: 
            a_s = -self.a_path_max # Phanh về đích
        elif self.v_s < v_path_adaptive - 0.05:
            a_s = self.a_path_max  # Tăng tốc đuổi theo (thoát cua)
        elif self.v_s > v_path_adaptive + 0.05:
            a_s = -self.a_path_max  # Phanh chủ động trước khi vào cua
        else:
            a_s = 0.0
        
        self.v_s = max(min(self.v_s + a_s * self.Ts, self.v_path_max), 0.02)
        self.s += self.v_s * self.Ts
        if self.s >= self.s_end: self.cmd_pub.publish(Twist()); return

        # B. REFERENCE & DERIVATIVES (Chain Rule)
        ss, vs, as_val = self.s, self.v_s, a_s
        f = self.freq
        
        x_ref = 1.1 + self.A * math.sin(f * ss)
        y_ref = 0.9 + self.A * math.sin(2 * f * ss)
        
        dx_ds = self.A * f * math.cos(f * ss)
        dy_ds = 2 * self.A * f * math.cos(2 * f * ss)
        
        dx_ref = dx_ds * vs
        dy_ref = dy_ds * vs
        
        d2x_ds2 = -self.A * (f**2) * math.sin(f * ss)
        d2y_ds2 = -4 * self.A * (f**2) * math.sin(2 * f * ss)
        
        ddx_ref = d2x_ds2 * (vs**2) + dx_ds * as_val
        ddy_ref = d2y_ds2 * (vs**2) + dy_ds * as_val

        # C. FEEDBACK LINEARIZATION LOGIC
        # ez = [pos_err; vel_err]
        phi = self.curr_q[2]
        ez1 = np.array([x_ref - self.curr_q[0], dx_ref - self.v_robot * math.cos(phi)])
        ez2 = np.array([y_ref - self.curr_q[1], dy_ref - self.v_robot * math.sin(phi)])
        
        K_final = self.K * min(t / 8.0, 1.0)
        # K_final = self.K 
        ux = ddx_ref + np.dot(K_final, ez1)
        uy = ddy_ref + np.dot(K_final, ez2)
        
        # Ma trận giải mã F (Decoupling Matrix)
        # Chống suy biến khi v gần bằng 0
        v_safe = max(abs(self.v_robot), 0.01) * np.sign(self.v_robot if self.v_robot != 0 else 1)
        
        # res(1) = v_dot, res(2) = w
        res = [0.0, 0.0]
        res[0] = ux * math.cos(phi) + uy * math.sin(phi)
        res[1] = (-ux * math.sin(phi) + uy * math.cos(phi)) / v_safe

        # D. CẬP NHẬT VẬN TỐC & RATE LIMIT
        # Giới hạn gia tốc dài v_accel
        v_accel = max(min(res[0], self.dv_max), -self.dv_max)
        # Tích phân vận tốc từ v_prev (giống v = v + v_accel*Ts)
        v_cmd = self.v_prev + v_accel * self.Ts
        v_final = max(min(v_cmd, self.v_max), -self.v_max)

        # Giới hạn vận tốc góc w và gia tốc góc delta_w
        w_target = max(min(res[1], self.w_max), -self.w_max)
        delta_w = max(min(w_target - self.w_prev, self.dw_max * self.Ts), -self.dw_max * self.Ts)
        w_final = self.w_prev + delta_w

        # Lưu lại cho bước sau
        self.v_prev = v_final
        self.w_prev = w_final

        # E. EXECUTE & LOG
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_final, w_final
        self.cmd_pub.publish(cmd)
        
        e_dist = math.sqrt((x_ref - self.curr_q[0])**2 + (y_ref - self.curr_q[1])**2)
        self.update_path_and_log(t, e_dist, v_final, w_final)

    def update_path_and_log(self, t, e_dist, v, w):
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.curr_q[0]
        pose.pose.position.y = self.curr_q[1]
        self.actual_path.poses.append(pose)
        if len(self.actual_path.poses) > 5000: self.actual_path.poses.pop(0)
        self.actual_path_pub.publish(self.actual_path)

        self.sum_e_dist_sq += (e_dist ** 2)
        self.count += 1
        rms_err = math.sqrt(self.sum_e_dist_sq / self.count)
        
        self.data_log.append([t, e_dist, rms_err, v, w])
        if int(t) > self.last_log_t:
            self.get_logger().info(f"Time: {t:.1f}s | Error: {e_dist:.3f}m | RMS_Err: {rms_err:.3f}m | v: {v:.3f}m/s | w: {w:.3f}rad/s")
            self.last_log_t = int(t)

    def save_csv(self):
        if self.data_log:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'Error (m)', 'RMS_Err (m)', 'v (m/s)', 'w (rad/s)'])
                writer.writerows(self.data_log)
            self.get_logger().info(f"Log saved: {self.csv_filename}")

def main():
    rclpy.init()
    node = FBLFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()