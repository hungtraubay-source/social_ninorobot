import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime
import numpy as np

class LyapunovFollower(Node):
    def __init__(self):
        super().__init__(
            'lyapunov_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # 1. PARAMETERS (Lyapunov Gains)
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('Kx', 3.0)
        self.declare_parameter('Ky', 80.0)
        self.declare_parameter('Kphi', 3.0)
        self.declare_parameter('num_loops',2)

        self.off_x = self.get_parameter('offset_x').value
        self.off_y = self.get_parameter('offset_y').value
        self.off_phi = self.get_parameter('offset_phi').value
        self.Ts = self.get_parameter('Ts').value
        self.Kx = self.get_parameter('Kx').value
        self.Ky = self.get_parameter('Ky').value
        self.Kphi = self.get_parameter('Kphi').value
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
        self.v_prev, self.w_prev = 0.0, 0.0
        
        self.v_max, self.w_max = 0.33, 3.85
        self.dv_max, self.dw_max = 1.0, 4.0

        self.start_time = None
        self.odom_received = False
        self.sum_e_dist_sq = 0.0
        self.count = 0
        self.last_log_t = -1
        self.data_log = []
        self.csv_filename = f'motion_law_lyapunov_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

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
        
        self.get_logger().info(f"Lyapunov Node Started. Play Gazebo!")

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
        self.odom_received = True

    def get_lookahead_safe_velocity(self, s_curr):
        """ Quét một đoạn look-ahead phía trước để tìm vận tốc an toàn nhỏ nhất """
        look_ahead_dist = 0.4  # Khoảng cách nhìn trước (m)
        num_samples = 5        # Số điểm lấy mẫu để quét
        v_min_found = self.v_path_max
        beta = 0.3             # Hệ số nhạy cảm với độ cong

        for i in range(num_samples + 1):
            # Tính vị trí s ở tương lai
            s_future = s_curr + (i / num_samples) * look_ahead_dist
            
            # Tính đạo hàm tại điểm s_future
            f = self.freq
            A = self.A
            dx = A * f * math.cos(f * s_future)
            dy = 2 * A * f * math.cos(2 * f * s_future)
            ddx = -A * f**2 * math.sin(f * s_future)
            ddy = -4 * A * f**2 * math.sin(2 * f * s_future)

            # Tính độ cong kappa
            v_ref_mag = math.sqrt(dx**2 + dy**2)
            kappa = abs(dx * ddy - dy * ddx) / (v_ref_mag**3 + 1e-6)
            
            # Vận tốc an toàn tại điểm này
            v_safe = self.v_path_max / (1.0 + beta * kappa)
            
            # Cập nhật giá trị nhỏ nhất
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
        
        if dist_to_end < 0.7:
            a_s = -self.a_path_max  # Phanh về đích
        elif self.v_s < v_path_adaptive - 0.05:
            a_s = self.a_path_max   # Tăng tốc/Duy trì
        elif self.v_s > v_path_adaptive + 0.05:
            a_s = -self.a_path_max  # Phanh trước khi vào cua
        else:
            a_s = 0.0
        
        self.v_s = max(min(self.v_s + a_s * self.Ts, self.v_path_max), 0.02)
        self.s += self.v_s * self.Ts
        if self.s >= self.s_end: self.cmd_pub.publish(Twist()); return

        # B. REFERENCE & DERIVATIVES (Chain Rule)
        ss, vs, as_val = self.s, self.v_s, a_s
        f = self.freq
        
        dx_ds = self.A * f * math.cos(f * ss)
        dy_ds = 2 * self.A * f * math.cos(2 * f * ss)
        dx_ref, dy_ref = dx_ds * vs, dy_ds * vs
        
        d2x_ds2 = -self.A * (f**2) * math.sin(f * ss)
        d2y_ds2 = -4 * self.A * (f**2) * math.sin(2 * f * ss)
        ddx_ref = d2x_ds2 * (vs**2) + dx_ds * as_val
        ddy_ref = d2y_ds2 * (vs**2) + dy_ds * as_val

        v_ref = math.sqrt(dx_ref**2 + dy_ref**2)
        w_ref = (dx_ref * ddy_ref - dy_ref * ddx_ref) / (dx_ref**2 + dy_ref**2 + 1e-6)
        phi_ref = math.atan2(dy_ref, dx_ref)

        # C. LYAPUNOV ERROR TRANSFORMATION
        # Global Error
        ex_g = 1.1 + self.A * math.sin(f * ss) - self.curr_q[0]
        ey_g = 0.9 + self.A * math.sin(2 * f * ss) - self.curr_q[1]
        ephi_g = math.atan2(math.sin(phi_ref - self.curr_q[2]), math.cos(phi_ref - self.curr_q[2]))

        # Transform to Robot Frame (Body Frame)
        cos_q = math.cos(self.curr_q[2])
        sin_q = math.sin(self.curr_q[2])
        e1 =  cos_q * ex_g + sin_q * ey_g
        e2 = -sin_q * ex_g + cos_q * ey_g
        e3 = ephi_g

        # D. LYAPUNOV CONTROL LAW
        # v_target = vRef*cos(e3) + Kx*e1
        # w_target = wRef + Ky*vRef*sinc(e3/pi)*e2 + Kphi*e3
        v_target = v_ref * math.cos(e3) + self.Kx * e1
        
        sinc_e3 = math.sin(e3) / e3 if abs(e3) > 1e-6 else 1.0
        w_target = w_ref + self.Ky * v_ref * sinc_e3 * e2 + self.Kphi * e3

        # E. RATE LIMIT & EXECUTE
        v_target = max(min(v_target, self.v_max), -self.v_max)
        delta_v = max(min(v_target - self.v_prev, self.dv_max * self.Ts), -self.dv_max * self.Ts)
        v_final = self.v_prev + delta_v

        w_target = max(min(w_target, self.w_max), -self.w_max)
        delta_w = max(min(w_target - self.w_prev, self.dw_max * self.Ts), -self.dw_max * self.Ts)
        w_final = self.w_prev + delta_w

        self.v_prev, self.w_prev = v_final, w_final

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_final, w_final
        self.cmd_pub.publish(cmd)
        
        e_dist = math.sqrt(ex_g**2 + ey_g**2)
        self.update_path_and_log(t, e_dist, v_final, w_final)

    def update_path_and_log(self, t, e_dist, v, w):
        pose = PoseStamped()
        pose.header.frame_id, pose.header.stamp = 'world', self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = self.curr_q[0], self.curr_q[1]
        self.actual_path.poses.append(pose)
        if len(self.actual_path.poses) > 5000: self.actual_path.poses.pop(0)
        self.actual_path_pub.publish(self.actual_path)

        self.sum_e_dist_sq += (e_dist ** 2)
        self.count += 1
        rms_err = math.sqrt(self.sum_e_dist_sq / self.count)
        
        self.data_log.append([t, e_dist, rms_err, v, w])
        if int(t) > self.last_log_t:
            self.get_logger().info(f"Time: {t:.1f}s | Error: {e_dist:.3f}m | RMS: {rms_err:.3f}m | v: {v:.3f}m/s | w: {w:.3f}rad/s")
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
    node = LyapunovFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()