import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime

class LyapunovFollower(Node):
    def __init__(self):
        super().__init__(
            'lyapunov_follower',
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
        
        # Lyapunov Parameters (zeta: damping, g: gain)
        self.zeta = 0.9
        self.g = 85.0
        
        self.v_max = 0.33
        self.w_max = 3.85
        
        self.start_time = None
        self.data_log = []
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'simulation_data_lyapunov_{now_str}.csv'

        # 2. PUBS/SUBS
        qos_path = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_path)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        self.last_log_t = -1.0
        self.sum_e_dist = 0.0
        self.count = 0

        self.create_timer(5.0, self.publish_static_ref_path)
        self.get_logger().info(f'Lyapunov Follower ready. Saving data to: {self.csv_filename}. Play Gazebo!')

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
            return
        
        t = (now - self.start_time).nanoseconds / 1e9

        # 1. Tọa độ Robot (Ma trận xoay xử lý offset)
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        o_phi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        cs, ss = math.cos(self.offset_phi), math.sin(self.offset_phi)
        curr_x = ox * cs - oy * ss + self.offset_x
        curr_y = ox * ss + oy * cs + self.offset_y
        curr_phi = math.atan2(math.sin(o_phi + self.offset_phi), math.cos(o_phi + self.offset_phi))

        # 2. Quỹ đạo tham chiếu & Đạo hàm
        f = self.freq
        xr = 1.1 + 0.7 * math.sin(f * t)
        yr = 0.9 + 0.7 * math.sin(2 * f * t)
        dxr = 0.7 * f * math.cos(f * t)
        dyr = 1.4 * f * math.cos(2 * f * t)
        ddxr = -0.7 * f**2 * math.sin(f * t)
        ddyr = -2.8 * f**2 * math.sin(2 * f * t)

        v_ref = math.sqrt(dxr**2 + dyr**2)
        w_ref = (dxr * ddyr - dyr * ddxr) / (dxr**2 + dyr**2 + 1e-6)
        phi_ref = math.atan2(dyr, dxr)

        # 3. CHUYỂN ĐỔI SAI SỐ SANG ROBOT FRAME
        ex = xr - curr_x
        ey = yr - curr_y
        e_phi = math.atan2(math.sin(phi_ref - curr_phi), math.cos(phi_ref - curr_phi))

        # e1: dọc, e2: ngang, e3: góc
        e1 = math.cos(curr_phi) * ex + math.sin(curr_phi) * ey
        e2 = -math.sin(curr_phi) * ex + math.cos(curr_phi) * ey
        e3 = e_phi

        # 4. LUẬT ĐIỀU KHIỂN LYAPUNOV
        Kx = 2 * self.zeta * math.sqrt(w_ref**2 + self.g * v_ref**2)
        Kphi = Kx
        Ky = self.g
        
        # Hàm sinc(x) = sin(x)/x
        sinc_e3 = math.sin(e3) / e3 if abs(e3) > 1e-6 else 1.0
        
        v = v_ref * math.cos(e3) + Kx * e1
        w = w_ref + Ky * v_ref * sinc_e3 * e2 + Kphi * e3

        # 5. Bão hòa & Gửi lệnh
        v = max(min(v, self.v_max), -self.v_max)
        w = max(min(w, self.w_max), -self.w_max)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        # 6. Log & CSV
        self.log_and_path(now, curr_x, curr_y, xr, yr, t, v, w)

    def log_and_path(self, now, cx, cy, xr, yr, t, cv, cw):
        ap = PoseStamped()
        ap.header.stamp, ap.header.frame_id = now.to_msg(), 'world'
        ap.pose.position.x, ap.pose.position.y = cx, cy
        self.actual_path.poses.append(ap)
        if len(self.actual_path.poses) > 2000: self.actual_path.poses.pop(0)
        self.actual_path_pub.publish(self.actual_path)

        e_dist = math.sqrt((xr - cx)**2 + (yr - cy)**2)
        self.sum_e_dist += e_dist
        self.count += 1
        s_avg = self.sum_e_dist / self.count
        self.data_log.append([t, e_dist, s_avg, float(cv), float(cw)])
        
        if int(t) > self.last_log_t:
            self.get_logger().info(f"Time: {t:.1f}s | Err: {e_dist:.3f}m | Avg Error: {s_avg:.3f}m | v: {cv:.3f}m/s | w: {cw:.3f}rad/s")
            self.last_log_t = int(t)

    def save_to_csv(self):
        if self.data_log:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'Error (m)', 'Avg Error (m)', 'v (m/s)', 'w (rad/s)'])
                writer.writerows(self.data_log)
            self.get_logger().info(f'Saved to {self.csv_filename}')

def main():
    rclpy.init()
    node = LyapunovFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.save_to_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()