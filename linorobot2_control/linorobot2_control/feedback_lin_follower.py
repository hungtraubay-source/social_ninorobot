import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime


class FeedbackLinearizationFollower(Node):
    def __init__(self):
        super().__init__(
            'fbl_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # 1. THÔNG SỐ ĐIỀU KHIỂN
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)

        self.offset_x = self.get_parameter('offset_x').get_parameter_value().double_value
        self.offset_y = self.get_parameter('offset_y').get_parameter_value().double_value
        self.offset_phi = self.get_parameter('offset_phi').get_parameter_value().double_value

        self.get_logger().info(f'Khởi tạo với Offset: x={self.offset_x}, y={self.offset_y}, phi={self.offset_phi}')

        self.freq = 2 * math.pi / 30
        
        # Gains từ poles -2 +/- 1i => s^2 + 4s + 5 = 0
        self.K1 = 5.0
        self.K2 = 4.0
        
        self.v_max = 0.33
        self.w_max = 3.85
        # self.v_max = 0.8
        # self.w_max = 9.0
        
        # Biến trạng thái
        self.start_time = None
        self.last_time = None
        self.v_internal = 0.0  # Bộ tích phân vận tốc (Velocity Integrator)
        
        # Lưu trữ để vẽ và log
        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        self.sum_e_dist = 0.0
        self.count = 0
        self.last_log_t = -1.0

        # Lưu trữ dữ liệu
        self.data_log = []  # Danh sách chứa dữ liệu
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'simulation_data_fbl_{now_str}.csv'

        # 2. QOS & PUBS/SUBS
        qos_path = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_path)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 3. VẼ QUỸ ĐẠO THAM CHIẾU
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.get_logger().info(f'FBL Follower ready. Saving data to: {self.csv_filename}. Play Gazebo!')

    def publish_static_ref_path(self):
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp = self.get_clock().now().to_msg()
        for i in range(301):
            tr = i * 0.1
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = 1.1 + 0.7 * math.sin(self.freq * tr)
            pose.pose.position.y = 0.9 + 0.7 * math.sin(2 * self.freq * tr)
            ref_path.poses.append(pose)
        self.ref_path_pub.publish(ref_path)

    def odom_callback(self, msg):
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            self.last_time = now
            return
        
        # Tính dt để tích phân
        dt = (now - self.last_time).nanoseconds / 1e9
        t = (now - self.start_time).nanoseconds / 1e9
        self.last_time = now
        if dt <= 0: return

        # 1. Tọa độ Robot (Ma trận xoay)
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        o_phi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        cs, ss = math.cos(self.offset_phi), math.sin(self.offset_phi)
        curr_x = ox * cs - oy * ss + self.offset_x
        curr_y = ox * ss + oy * cs + self.offset_y
        curr_phi = math.atan2(math.sin(o_phi + self.offset_phi), math.cos(o_phi + self.offset_phi))

        # 2. Reference & Derivatives (Đạo hàm cấp 1, cấp 2)
        f = self.freq
        x_ref = 1.1 + 0.7 * math.sin(f * t)
        y_ref = 0.9 + 0.7 * math.sin(2 * f * t)
        dx_ref = 0.7 * f * math.cos(f * t)
        dy_ref = 1.4 * f * math.cos(2 * f * t)
        ddx_ref = -0.7 * f**2 * math.sin(f * t)
        ddy_ref = -2.8 * f**2 * math.sin(2 * f * t)

        # 3. Trạng thái hiện tại (z1, z2)
        # x_dot = v * cos(phi), y_dot = v * sin(phi)
        vx_curr = self.v_internal * math.cos(curr_phi)
        vy_curr = self.v_internal * math.sin(curr_phi)

        # 4. Tính toán Luật điều khiển (u = dd_ref + K*e)
        ux = ddx_ref + self.K1 * (x_ref - curr_x) + self.K2 * (dx_ref - vx_curr)
        uy = ddy_ref + self.K1 * (y_ref - curr_y) + self.K2 * (dy_ref - vy_curr)

        # 5. Inverse Kinematics (Feedback Linearization Transformation)
        # Tránh chia cho 0 khi v quá nhỏ
        v_eff = max(abs(self.v_internal), 0.01) * (1 if self.v_internal >= 0 else -1)
        
        v_dot = ux * math.cos(curr_phi) + uy * math.sin(curr_phi)
        w = (-math.sin(curr_phi) * ux + math.cos(curr_phi) * uy) / v_eff

        # 6. Tích phân vận tốc & Bão hòa
        self.v_internal += v_dot * dt
        self.v_internal = max(min(self.v_internal, self.v_max), -self.v_max)
        w = max(min(w, self.w_max), -self.w_max)

        # 7. Publish
        cmd = Twist()
        cmd.linear.x = self.v_internal
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        # 8. Path & Log
        self.update_path_and_log(now, curr_x, curr_y, x_ref, y_ref, t, self.v_internal, w)

    def update_path_and_log(self, now, cx, cy, xr, yr, t, cv, cw):
        actual_pose = PoseStamped()
        actual_pose.header.stamp = now.to_msg()
        actual_pose.header.frame_id = 'world'
        actual_pose.pose.position.x, actual_pose.pose.position.y = cx, cy
        self.actual_path.poses.append(actual_pose)
        if len(self.actual_path.poses) > 2000: self.actual_path.poses.pop(0)
        self.actual_path_pub.publish(self.actual_path)

        e_dist = math.sqrt((xr - cx)**2 + (yr - cy)**2)
        self.sum_e_dist += e_dist
        self.count += 1
        s_average = self.sum_e_dist / self.count
        self.data_log.append([t, e_dist, s_average, float(cv), float(cw)])
        if int(t) > self.last_log_t:
            self.get_logger().info(f"Time: {t:.1f}s | Err: {e_dist:.3f}m | Avg Error: {s_average:.3f}m | v: {cv:.3f}m/s | w: {cw:.3f}rad/s")
            self.last_log_t = int(t)

    def save_to_csv(self):
        if not self.data_log:
            self.get_logger().info("Không có dữ liệu để lưu.")
            return

        with open(self.csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Viết tiêu đề cột cho Excel
            writer.writerow(['Time (s)', 'Error (m)', 'Avg Error (m)', 'v (m/s)', 'w (rad/s)'])
            # Viết toàn bộ dữ liệu
            writer.writerows(self.data_log)
        
        self.get_logger().info(f'Saved to {self.csv_filename}')

def main(args=None):
    rclpy.init(args=args)
    node = FeedbackLinearizationFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Đang dừng node và lưu dữ liệu...')
    finally:
        node.save_to_csv() # Gọi hàm lưu file trước khi hủy node
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()