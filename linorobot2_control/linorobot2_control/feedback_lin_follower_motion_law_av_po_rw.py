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
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)
            ]
        )

        # 1. PARAMETERS
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('Kp', 1)
        self.declare_parameter('Kd', 1)
        self.declare_parameter('num_loops', 10)

        self.off_x = self.get_parameter('offset_x').value
        self.off_y = self.get_parameter('offset_y').value
        self.off_phi = self.get_parameter('offset_phi').value
        self.Ts = self.get_parameter('Ts').value
        self.K = np.array([self.get_parameter('Kd').value, self.get_parameter('Kp').value])
        self.num_loops = self.get_parameter('num_loops').value

        self.get_logger().info(f"REAL ROBOT: Ts={self.Ts}s. Offset: {self.off_x}, {self.off_y}, {self.off_phi}")

        # 2. TRAJECTORY CONSTANTS
        self.A = 0.7
        self.s_end_one_loop = 30.0
        self.s_end = self.s_end_one_loop * self.num_loops
        self.freq = 2 * math.pi / self.s_end_one_loop
        self.v_path_max = 0.8
        self.a_path_max = 0.5
        self.L = 0.05

        # 3. STATE VARIABLES
        self.s = 0.0
        self.v_s = 0.01
        self.curr_q = [self.off_x, self.off_y, self.off_phi]
        self.v_robot = 0.01 # Vận tốc dài hiện tại
        self.v_prev, self.w_prev = 0.0, 0.0
        
        self.v_max, self.w_max = 0.33, 3.85
        self.dv_max, self.dw_max = 1.0, 4.0

        # Modified
        self.L = 0.08
        self.Kp_final = 0.5
        self.v_max = 0.2
        self.w_max = 2.0
        self.dv_max = 1.0
        self.dw_max = 4.0
        self.v_path_max = 0.4
        self.a_path_max = 0.2

        self.start_time = None
        self.odom_received = False
        self.sum_e_dist_sq = 0.0
        self.count = 0
        self.last_log_t = -1
        self.data_log = []
        self.csv_filename = f'real_motion_law_fbl_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

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
        
        self.get_logger().info(f"FBL Follower initialized.")

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
        look_ahead_dist = 0.4  # Nhìn trước n mét
        num_samples = 5        # Số điểm lấy mẫu
        beta = 0.01             # Hệ số nhạy cảm (Tăng beta nếu muốn cua chậm hơn)
        
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
        
        if dist_to_end < 1.0: 
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
        ss, vs = self.s, self.v_s
        f = self.freq
        
        x_ref = 1.1 + self.A * math.sin(f * ss)
        y_ref = 0.9 + self.A * math.sin(2 * f * ss)
        
        dx_ds = self.A * f * math.cos(f * ss)
        dy_ds = 2 * self.A * f * math.cos(2 * f * ss)
        
        dx_ref = dx_ds * vs
        dy_ref = dy_ds * vs

        # C. FEEDBACK LINEARIZATION LOGIC
        # 1. Tọa độ điểm nhìn trước P
        L = self.L # Khoảng cách offset
        phi = self.curr_q[2]
        xp = self.curr_q[0] + L * math.cos(phi)
        yp = self.curr_q[1] + L * math.sin(phi)

        # 2. Vận tốc tham chiếu (Chỉ cần đạo hàm bậc 1)
        # dx_ref, dy_ref đã tính từ vs và s
        
        # 3. Luật điều khiển P (Proportional Control)
        # ux, uy ở đây chính là vận tốc mong muốn của điểm P
        Kp_pos = self.Kp_final * min(t / 10.0, 1.0)
        ux = dx_ref + Kp_pos * (x_ref - xp)
        uy = dy_ref + Kp_pos * (y_ref - yp)

        # 4. Ma trận giải mã trực tiếp ra v và w (Không thông qua tích phân)
        v_target = ux * math.cos(phi) + uy * math.sin(phi)
        w_target = (-ux * math.sin(phi) + uy * math.cos(phi)) / L

        # D. CẬP NHẬT VẬN TỐC & RATE LIMIT
        v_target = max(min(v_target, self.v_max), -self.v_max)
        delta_v = max(min(v_target - self.v_prev, self.dv_max * self.Ts), -self.dv_max * self.Ts)
        v_final = self.v_prev + delta_v

        # Giới hạn vận tốc góc w và gia tốc góc delta_w
        w_target = max(min(w_target, self.w_max), -self.w_max)
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
        if len(self.actual_path.poses) > 2000: self.actual_path.poses.pop(0)
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