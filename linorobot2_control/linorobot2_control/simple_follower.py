import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__(
            'simple_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # 1. KHAI BÁO THÔNG SỐ
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)

        self.offset_x = self.get_parameter('offset_x').get_parameter_value().double_value
        self.offset_y = self.get_parameter('offset_y').get_parameter_value().double_value
        self.offset_phi = self.get_parameter('offset_phi').get_parameter_value().double_value

        self.get_logger().info(f'Khởi tạo với Offset: x={self.offset_x}, y={self.offset_y}, phi={self.offset_phi}')

        self.freq = 2 * math.pi / 30
        self.Kv = 3.0
        self.Kphi = 4.0
        self.v_max = 0.33
        self.w_max = 3.85
        # self.v_max, self.w_max = 0.8, 9.0
        
        # Biến quản lý thời gian và sai số
        self.start_time = None
        self.max_path_points = 1000
        self.sum_e_dist = 0.0
        self.count = 0
        self.last_log_t = -1.0

        # Lưu trữ dữ liệu
        self.data_log = []  # Danh sách chứa dữ liệu
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'simulation_data_simple_{now_str}.csv'

        # 2. THIẾT LẬP QOS CHO REF PATH TĨNH (Đường đỏ)
        # Transient Local giúp RViz "nhớ" được đường này kể cả khi bật sau
        qos_path = QoSProfile(depth=1)
        qos_path.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # 3. PUBLISHERS & SUBSCRIBERS
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_path)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'

        # 4. VẼ QUỸ ĐẠO THAM CHIẾU
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.get_logger().info(f'Simple Follower ready. Saving data to: {self.csv_filename}. Play Gazebo!')
        
    def publish_static_ref_path(self):
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp = self.get_clock().now().to_msg()
        
        # Vẽ chu kỳ 30 giây của hình sin
        for i in range(300):
            t_ref = i * 0.1
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = 1.1 + 0.7 * math.sin(self.freq * t_ref)
            pose.pose.position.y = 0.9 + 0.7 * math.sin(2 * self.freq * t_ref)
            ref_path.poses.append(pose)
            
        self.ref_path_pub.publish(ref_path)

    def odom_callback(self, msg):
    	### Tracking Control ###
        # Đồng bộ thời gian với Gazebo
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            return
        
        # 1. Tính thời gian trôi qua (t)
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9

        # 2. Cập nhật vị trí robot
        # Lấy dữ liệu thô từ Odom (tương đối so với điểm spawn)
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        
        q = msg.pose.pose.orientation
        odom_phi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        # Dùng ma trận xoay để chuyển sang hệ world
        cos_s = math.cos(self.offset_phi)
        sin_s = math.sin(self.offset_phi)

        curr_x = odom_x * cos_s - odom_y * sin_s + self.offset_x
        curr_y = odom_x * sin_s + odom_y * cos_s + self.offset_y
        
        # Góc phi trong world
        curr_phi = odom_phi + self.offset_phi
        curr_phi = math.atan2(math.sin(curr_phi), math.cos(curr_phi)) # Chuẩn hóa góc

        # 3. Quỹ đạo tham chiếu (Reference)
        x_ref = 1.1 + 0.7 * math.sin(self.freq * t)
        y_ref = 0.9 + 0.7 * math.sin(2 * self.freq * t)
        
        # Góc tham chiếu dựa trên hướng đi tới điểm tiếp theo
        phi_ref = math.atan2(y_ref - curr_y, x_ref - curr_x)

        # 4. Tính toán sai số (Error)
        e_dist = math.sqrt((x_ref - curr_x)**2 + (y_ref - curr_y)**2)
        e_phi = phi_ref - curr_phi
        # Chuẩn hóa góc về [-pi, pi]
        e_phi = math.atan2(math.sin(e_phi), math.cos(e_phi))

        # 5. Bộ điều khiển Upgraded Controller
        v = self.Kv * e_dist
        
        # Logic đi lùi nếu góc lệch quá lớn (> 90 độ)
        v = v * math.cos(e_phi) 
        e_phi = math.atan(math.tan(e_phi)) 
        w = self.Kphi * e_phi

        # 6. Giới hạn (Saturation)
        v = max(min(v, self.v_max), -self.v_max)
        w = max(min(w, self.w_max), -self.w_max)

        # 7. Gửi lệnh
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        
        ### Vẽ quỹ đạo ###
        stamp = self.get_clock().now().to_msg()

        # Vẽ quỹ đạo thực tế (Đường màu xanh)
        actual_pose = PoseStamped()
        actual_pose.header.stamp = stamp
        actual_pose.header.frame_id = 'world'
        actual_pose.pose.position.x = curr_x
        actual_pose.pose.position.y = curr_y
        self.actual_path.poses.append(actual_pose)
        
        # Giới hạn độ dài
        if len(self.actual_path.poses) > self.max_path_points:
            self.actual_path.poses.pop(0)

        self.actual_path_pub.publish(self.actual_path)
        
        ### Print Error ###
        # 1. Tính toán sai số trung bình (S_average)
        self.sum_e_dist += e_dist
        self.count += 1
        s_average = self.sum_e_dist / self.count
        self.data_log.append([t, e_dist, s_average, v, w])

        # 2. In kiểm tra Error ra Terminal (Mỗi 1 giây in 1 lần cho đỡ mỏi mắt)
        if int(t) > self.last_log_t:
            self.get_logger().info(
                f"Time: {t:.1f}s | Error: {e_dist:.3f}m | Avg Error: {s_average:.3f}m | v: {v:.3f}m/s | w: {w:.3f}rad/s"
            )
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
    node = TrajectoryFollower()
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
