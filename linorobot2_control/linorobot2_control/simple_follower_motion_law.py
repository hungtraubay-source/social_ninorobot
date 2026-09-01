import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime
import sys

class TrajectoryFollowerFull(Node):
    def __init__(self):
        # 1. ĐỒNG BỘ SIM TIME VÀ KHỞI TẠO NODE
        super().__init__(
            'simple_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # 2. KHAI BÁO PARAMETERS
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('Kv', 1.0)
        self.declare_parameter('Kphi', 2.0)
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

        self.get_logger().info(f"Ts={self.Ts}s. Offset: {self.offset_x}, {self.offset_y}, {self.offset_phi}")

        # 3. THÔNG SỐ QUỸ ĐẠO HÌNH SỐ 8 (S(t) Motion Law)
        self.s = 0.0
        self.v_s = 0.01
        self.s_end = 30.0
        self.s_end_ref = self.s_end
        self.freq = 2 * math.pi / self.s_end
        self.A = 0.7
        self.v_path_max = 0.8
        self.a_path_max = 0.5
        self.s_end = self.num_loops * self.s_end

        self.start_time = None
        self.sum_e_dist = 0.0
        self.count = 0
        self.last_log_t = -1.0

        # 4. GIỚI HẠN GIA TỐC
        self.v_max, self.w_max = 0.33, 3.85
        self.dv_max, self.dw_max = 1.0, 4.0
        self.v_prev, self.w_prev = 0.0, 0.0

        # 5. BIẾN TRẠNG THÁI
        self.curr_q = [self.offset_x, self.offset_y, self.offset_phi]
        self.odom_received = False
        self.data_log = []
        self.csv_filename = f'motion_law_data_simple_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # 6. PUBLISHERS & SUBSCRIBERS
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # QoS cho đường dẫn tĩnh (đường đỏ)
        qos_path = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_path)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 7. KHỞI TẠO ĐƯỜNG DẪN
        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        
        # 8. TIMER ĐIỀU KHIỂN CHÍNH
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.timer = self.create_timer(self.Ts, self.control_loop)
        
        self.get_logger().info(f"Node Full initialized. Play Gazebo!")

    def publish_static_ref_path(self):
        """ Vẽ đường hình số 8 tĩnh trên RViz (Đường màu đỏ) """
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp = self.get_clock().now().to_msg()
        
        for i in range(400):
            s_val = (i / 400.0) * self.s_end_ref
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            pose.pose.position.x = 1.1 + self.A * math.sin(self.freq * s_val)
            pose.pose.position.y = 0.9 + self.A * math.sin(2 * self.freq * s_val)
            ref_path.poses.append(pose)
        self.ref_path_pub.publish(ref_path)

    def odom_callback(self, msg):
        """ Cập nhật vị trí robot về hệ tọa độ World """
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ophi = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        c, s = math.cos(self.offset_phi), math.sin(self.offset_phi)
        self.curr_q[0] = ox * c - oy * s + self.offset_x
        self.curr_q[1] = ox * s + oy * c + self.offset_y
        self.curr_q[2] = math.atan2(math.sin(ophi + self.offset_phi), math.cos(ophi + self.offset_phi))
        self.odom_received = True

    def control_loop(self):
        if not self.odom_received: return

        # Lấy sim_time hiện tại
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            return
        
        t = (now - self.start_time).nanoseconds / 1e9

        # A. MOTION LAW s(t)
        dist_to_end = self.s_end - self.s
        if self.s < 1.0: a_s = self.a_path_max
        elif dist_to_end < 1.0: a_s = -self.a_path_max
        else: a_s = 0.0
        
        self.v_s = max(min(self.v_s + a_s * self.Ts, self.v_path_max), 0.05)
        self.s += self.v_s * self.Ts

        if self.s >= self.s_end:
            self.cmd_pub.publish(Twist())
            return

        # B. REFERENCE POSE
        x_ref = 1.1 + self.A * math.sin(self.freq * self.s)
        y_ref = 0.9 + self.A * math.sin(2 * self.freq * self.s)
        phi_ref = math.atan2(y_ref - self.curr_q[1], x_ref - self.curr_q[0])

        # C. ERROR & CONTROL (P-Controller Upgraded)
        e_dist = math.sqrt((x_ref - self.curr_q[0])**2 + (y_ref - self.curr_q[1])**2)
        e_phi = math.atan2(math.sin(phi_ref - self.curr_q[2]), math.cos(phi_ref - self.curr_q[2]))

        v_raw = self.Kv * e_dist
        w_raw = self.Kphi * e_phi

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

        # E. PUBLISH & LOGGING
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_final, w_final
        self.cmd_pub.publish(cmd)
        self.v_prev, self.w_prev = v_final, w_final

        self.update_path_and_log(t, e_dist, v_final, w_final)

    def update_path_and_log(self, t, e_dist, v, w):
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.curr_q[0]
        pose.pose.position.y = self.curr_q[1]
        
        self.actual_path.poses.append(pose)
        if len(self.actual_path.poses) > 2000:
            self.actual_path.poses.pop(0)
        self.actual_path_pub.publish(self.actual_path)

        self.sum_e_dist += (e_dist ** 2)
        self.count += 1
        rms_error = math.sqrt(self.sum_e_dist / self.count)

        self.data_log.append([t, e_dist, rms_error, v, w])

        if int(t) > self.last_log_t:
            self.get_logger().info(
                f"Time: {t:.1f}s | Error: {e_dist:.3f}m | RMS_Err: {rms_error:.3f}m | v: {v:.3f}m/s | w: {w:.3f}rad/s"
            )
            self.last_log_t = int(t)

    def save_csv(self):
        if self.data_log:
            with open(self.csv_filename, 'w', newline='') as f:
                csv.writer(f).writerows([['Time (s)', 'Error (m)', 'RMS_Err (m)', 'v (m/s)', 'w (rad/s)']] + self.data_log)
            self.get_logger().info(f"Log saved: {self.csv_filename}")

def main():
    rclpy.init()
    node = TrajectoryFollowerFull()
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