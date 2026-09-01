import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime

class RealRobotFollower(Node):
    def __init__(self):
        # 1. DÙNG THỜI GIAN THỰC (Real Time)
        super().__init__(
            'simple_follower_real',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)
            ]
        )

        # 2. KHAI BÁO PARAMETERS
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('Kv', 2.0) 
        self.declare_parameter('Kphi', 3.0)
        self.declare_parameter('upgradedControl', False)
        self.declare_parameter('num_loops', 10)

        self.offset_x = self.get_parameter('offset_x').value
        self.offset_y = self.get_parameter('offset_y').value
        self.offset_phi = self.get_parameter('offset_phi').value
        self.Ts = self.get_parameter('Ts').value
        self.Kv = self.get_parameter('Kv').value
        self.Kphi = self.get_parameter('Kphi').value
        self.upgradedControl = self.get_parameter('upgradedControl').value
        self.num_loops = self.get_parameter('num_loops').value

        self.get_logger().info(f"REAL ROBOT: Ts={self.Ts}s. Offset: {self.offset_x}, {self.offset_y}, {self.offset_phi}")

        # 3. THÔNG SỐ QUỸ ĐẠO HÌNH SỐ 8
        self.s = 0.0
        self.v_s = 0.01
        self.s_one_loop = 30.0
        self.s_end = self.num_loops * self.s_one_loop
        self.freq = 2 * math.pi / self.s_one_loop
        self.A = 0.7
        self.v_path_max = 0.8
        self.a_path_max = 0.5

        # 4. GIỚI HẠN GIA TỐC
        self.v_max = 0.33
        self.w_max = 3.85
        self.dv_max = 1.0
        self.dw_max = 4.0
        self.v_prev, self.w_prev = 0.0, 0.0

        # REAL ROBOT
        self.Kv = 0.5
        self.Kphi = 1.0
        self.v_max = 0.3
        self.w_max = 3.0
        self.dv_max = 1.0
        self.dw_max = 4.0
        self.v_path_max = 0.6
        self.a_path_max = 0.3

        # 5. BIẾN TRẠNG THÁI
        self.curr_q = [self.offset_x, self.offset_y, self.offset_phi]
        self.odom_received = False
        self.start_time = None
        self.sum_e_dist_sq = 0.0
        self.count = 0
        self.last_log_t = -1.0
        self.data_log = []
        self.csv_filename = f'real_motion_law_simple_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # 6. ROS INFRASTRUCTURE
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos_path = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_path)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 7. KHỞI TẠO PATH (Dùng frame world)
        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        
        # 8. TIMERS
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.timer = self.create_timer(self.Ts, self.control_loop)

        self.get_logger().info(f'Simple Follower ready.')

    def publish_static_ref_path(self):
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp = self.get_clock().now().to_msg()
        for i in range(400):
            s_val = (i / 400.0) * self.s_one_loop
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = 1.1 + self.A * math.sin(self.freq * s_val)
            pose.pose.position.y = 0.9 + self.A * math.sin(2 * self.freq * s_val)
            ref_path.poses.append(pose)
        self.ref_path_pub.publish(ref_path)

    def odom_callback(self, msg):
        # Lấy dữ liệu thô từ odom (odom frame)
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ophi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        # Áp dụng ma trận xoay và offset để đưa về world frame
        c = math.cos(self.offset_phi)
        s = math.sin(self.offset_phi)
        
        self.curr_q[0] = ox * c - oy * s + self.offset_x
        self.curr_q[1] = ox * s + oy * c + self.offset_y
        self.curr_q[2] = math.atan2(math.sin(ophi + self.offset_phi), math.cos(ophi + self.offset_phi))
        self.odom_received = True

    def control_loop(self):
        if not self.odom_received: return
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            return
        t = (now - self.start_time).nanoseconds / 1e9

        # --- LOOK-AHEAD ADAPTIVE VELOCITY ---
        look_ahead_dist = 0.4
        num_samples = 5        # Chia làm 5 điểm để quét
        
        # Giả định vận tốc an toàn nhất là vận tốc max ban đầu
        v_path_adaptive = self.v_path_max
        beta = 0.1

        for i in range(num_samples + 1):
            # Tính vị trí s ở tương lai
            s_future = self.s + (i / num_samples) * look_ahead_dist
            
            # Tính đạo hàm tại điểm tương lai
            f = self.freq
            A = self.A
            dx_f = A * f * math.cos(f * s_future)
            dy_f = 2 * A * f * math.cos(2 * f * s_future)
            ddx_f = -A * f**2 * math.sin(f * s_future)
            ddy_f = -4 * A * f**2 * math.sin(2 * f * s_future)

            # Độ cong tại điểm tương lai
            v_ref_mag_f = math.sqrt(dx_f**2 + dy_f**2)
            kappa_f = abs(dx_f * ddy_f - dy_f * ddx_f) / (v_ref_mag_f**3 + 1e-6)
            
            # Vận tốc an toàn tại điểm tương lai đó
            v_safe_future = self.v_path_max / (1.0 + beta * kappa_f)
            
            # Lấy giá trị nhỏ nhất (Nếu tương lai có cua gắt, ta phải chậm từ bây giờ)
            if v_safe_future < v_path_adaptive:
                v_path_adaptive = v_safe_future
        # ------------------------------------

        # A. MOTION LAW s(t)
        dist_to_end = self.s_end - self.s
        
        if dist_to_end < 0.5:
            a_s = -self.a_path_max # Phanh về đích
        elif self.v_s < v_path_adaptive - 0.05:
            a_s = self.a_path_max  # Tăng tốc đuổi theo
        elif self.v_s > v_path_adaptive + 0.05:
            a_s = -self.a_path_max # Phanh trước cua
        else:
            a_s = 0.0
        
        self.v_s = max(min(self.v_s + a_s * self.Ts, self.v_path_max), 0.05)
        self.s += self.v_s * self.Ts

        if self.s >= self.s_end:
            self.cmd_pub.publish(Twist())
            return

        # B. REFERENCE POSE (Trong world frame)
        x_ref = 1.1 + self.A * math.sin(self.freq * self.s)
        y_ref = 0.9 + self.A * math.sin(2 * self.freq * self.s)
        phi_ref = math.atan2(y_ref - self.curr_q[1], x_ref - self.curr_q[0])

        # C. ERROR & CONTROL
        e_dist = math.sqrt((x_ref - self.curr_q[0])**2 + (y_ref - self.curr_q[1])**2)
        e_phi = math.atan2(math.sin(phi_ref - self.curr_q[2]), math.cos(phi_ref - self.curr_q[2]))

        Kv = self.Kv * min(t / 3.0, 1.0)
        Kphi = self.Kphi * min(t / 3.0, 1.0)
        v_raw = Kv * e_dist
        w_raw = Kphi * e_phi

        if self.upgradedControl:
            v_raw = self.Kv * e_dist * math.copysign(1.0, math.cos(e_phi))
            e_phi_mapped = math.atan(math.tan(e_phi)) 
            w_raw = self.Kphi * e_phi_mapped

        # D. RATE LIMIT
        v_target = max(min(v_raw, self.v_max), -self.v_max)
        dv = max(min(v_target - self.v_prev, self.dv_max * self.Ts), -self.dv_max * self.Ts)
        v_final = self.v_prev + dv

        w_target = max(min(w_raw, self.w_max), -self.w_max)
        dw = max(min(w_target - self.w_prev, self.dw_max * self.Ts), -self.dw_max * self.Ts)
        w_final = self.w_prev + dw

        # E. PUBLISH
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_final, w_final
        self.cmd_pub.publish(cmd)
        self.v_prev, self.w_prev = v_final, w_final

        self.update_path_and_log(t, e_dist, v_final, w_final)

    def update_path_and_log(self, t, e_dist, v, w):
        pose = PoseStamped()
        pose.header.frame_id, pose.header.stamp = 'world', self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = self.curr_q[0], self.curr_q[1]
        self.actual_path.poses.append(pose)
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
            self.get_logger().info(f"Log saved: {self.csv_filename}")

def main():
    rclpy.init()
    node = RealRobotFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()