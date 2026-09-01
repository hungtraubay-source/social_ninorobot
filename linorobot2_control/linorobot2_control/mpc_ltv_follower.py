import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
import math
import csv
import datetime

class MPCLTVFollower(Node):
    def __init__(self):
        super().__init__(
            'mpc_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )

        # 1. THÔNG SỐ ĐIỀU KHIỂN
        self.Ts = 0.1     # Sampling time
        self.N = 8          # Horizon (Nhìn trước N bước)
        self.v_max = 0.33   # Giới hạn theo robot thật
        self.w_max = 3.85
        self.ar = 0.65      # Reference error dynamics gain
        
        # Trọng số tối ưu (Qt: Error, Rt: Control effort)
        self.Q_diag = np.array([100.0, 100.0, 0.1])
        self.R_diag = np.array([0.1, 0.1])
        
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)

        self.offset_x = self.get_parameter('offset_x').get_parameter_value().double_value
        self.offset_y = self.get_parameter('offset_y').get_parameter_value().double_value
        self.offset_phi = self.get_parameter('offset_phi').get_parameter_value().double_value

        self.get_logger().info(f'Khởi tạo với Offset: x={self.offset_x}, y={self.offset_y}, phi={self.offset_phi}')

        self.freq = 2 * math.pi / 30

        # 2. LOGGING & PATH
        self.start_time = None
        self.data_log = []
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f'simulation_data_mpc_{now_str}.csv'
        
        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        self.sum_e_dist = 0.0
        self.count = 0
        self.last_log_t = -1.0

        # 3. ROS PUBS/SUBS
        qos_path = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_path)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.create_timer(5.0, self.publish_static_ref_path)
        self.get_logger().info(f'MPC-LTV Follower ready. Saving data to: {self.csv_filename}. Play Gazebo!')

    def get_ref_state(self, t_val):
        """Tính toán q_ref và u_ref tại thời điểm t"""
        xr = 1.1 + 0.7 * math.sin(self.freq * t_val)
        yr = 0.9 + 0.7 * math.sin(2 * self.freq * t_val)
        dxr = 0.7 * self.freq * math.cos(self.freq * t_val)
        dyr = 1.4 * self.freq * math.cos(2 * self.freq * t_val)
        ddxr = -0.7 * self.freq**2 * math.sin(self.freq * t_val)
        ddyr = -2.8 * self.freq**2 * math.sin(2 * self.freq * t_val)
        
        v_ref = math.sqrt(dxr**2 + dyr**2)
        w_ref = (dxr * ddyr - dyr * ddxr) / (dxr**2 + dyr**2 + 1e-6)
        phi_ref = math.atan2(dyr, dxr)
        return np.array([xr, yr, phi_ref]), np.array([v_ref, w_ref])

    def publish_static_ref_path(self):
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp = self.get_clock().now().to_msg()
        for i in range(301):
            tr = i * 0.1
            pose = PoseStamped()
            pose.header.frame_id = 'world'
            q_r, _ = self.get_ref_state(tr)
            pose.pose.position.x, pose.pose.position.y = q_r[0], q_r[1]
            ref_path.poses.append(pose)
        self.ref_path_pub.publish(ref_path)

    def odom_callback(self, msg):
        now = self.get_clock().now()
        if self.start_time is None:
            self.start_time = now
            return
        
        t = (now - self.start_time).nanoseconds / 1e9

        # --- 1. LẤY TRẠNG THÁI HIỆN TẠI (q) ---
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        o_phi = math.atan2(2*(quat.w*quat.z + quat.x*quat.y), 1 - 2*(quat.y*quat.y + quat.z*quat.z))
        
        # Ma trận xoay xử lý offset
        cs, ss = math.cos(self.offset_phi), math.sin(self.offset_phi)
        curr_x = ox * cs - oy * ss + self.offset_x
        curr_y = ox * ss + oy * cs + self.offset_y
        curr_phi = math.atan2(math.sin(o_phi + self.offset_phi), math.cos(o_phi + self.offset_phi))
        curr_q = np.array([curr_x, curr_y, curr_phi])

        # --- 2. TÍNH SAI SỐ TRONG ROBOT FRAME (e) ---
        q_ref0, u_ref0 = self.get_ref_state(t)
        rot_m = np.array([[math.cos(curr_phi),  math.sin(curr_phi), 0],
                          [-math.sin(curr_phi), math.cos(curr_phi), 0],
                          [0, 0, 1]])
        e = rot_m @ (q_ref0 - curr_q)
        e[2] = math.atan2(math.sin(e[2]), math.cos(e[2])) # wrapToPi

        # --- 3. XÂY DỰNG MA TRẬN Hm, Fm (LTV Prediction) ---
        # Dùng A0, A1, A2, A3... trong MATLAB để xây dựng quan hệ sai số tương lai
        B = np.array([[self.Ts, 0], [0, 0], [0, self.Ts]])
        C = np.eye(3)
        
        A_list = []
        for i in range(self.N + 1):
            _, u_ri = self.get_ref_state(t + i * self.Ts)
            Ai = np.array([[1, self.Ts * u_ri[1], 0],
                           [-self.Ts * u_ri[1], 1, self.Ts * u_ri[0]],
                           [0, 0, 1]])
            A_list.append(Ai)

        # Tính Fm (Prediction of free response)
        Fm = []
        term_f = C
        for i in range(self.N):
            term_f = term_f @ A_list[i]
            Fm.append(term_f)
        Fm = np.vstack(Fm)

        # Tính Hm (System Matrix - Tương tác giữa input và error tương lai)
        Hm = np.zeros((3 * self.N, 2 * self.N))
        for i in range(self.N):
            for j in range(i + 1):
                prod_A = C
                for k in range(j, i):
                    prod_A = prod_A @ A_list[k]
                Hm[i*3:(i+1)*3, j*2:(j+1)*2] = prod_A @ B

        # --- 4. TỐI ƯU HÓA (Giải thuật Least Squares) ---
        Qt = np.kron(np.eye(self.N), np.diag(self.Q_diag))
        Rt = np.kron(np.eye(self.N), np.diag(self.R_diag))
        
        # Công thức giải tích: KK = (Hm' * Qt * Hm + Rt) \ (Hm' * Qt * (-Fm))
        try:
            inv_part = np.linalg.inv(Hm.T @ Qt @ Hm + Rt)
            KKgpc = inv_part @ (Hm.T @ Qt @ (-Fm))
            KK = KKgpc[0:2, :] # Lấy gain cho bước hiện tại
        except np.linalg.LinAlgError:
            self.get_logger().error("Singular Matrix! MPC Solver failed.")
            return

        # Tính toán vận tốc (Feedback + Feedforward)
        v_feedback = -KK @ e
        uF = np.array([u_ref0[0] * math.cos(e[2]), u_ref0[1]])
        u = v_feedback + uF

        # --- 5. SATURATION & PUBLISH ---
        u[0] = np.clip(u[0], -self.v_max, self.v_max)
        u[1] = np.clip(u[1], -self.w_max, self.w_max)

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(u[0]), float(u[1])
        self.cmd_pub.publish(cmd)

        # --- 6. LOGGING & PATH ---
        self.update_log_and_path(now, curr_x, curr_y, q_ref0[0], q_ref0[1], t, u)

    def update_log_and_path(self, now, cx, cy, xr, yr, t, u):
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
        self.data_log.append([t, e_dist, s_avg, float(u[0]), float(u[1])])
        
        if int(t) > self.last_log_t:
            self.get_logger().info(f"Time: {t:.1f}s | Err: {e_dist:.3f}m | Avg: {s_avg:.3f}m | v: {u[0]:.3f}m/s | w: {u[1]:.3f}rad/s")
            self.last_log_t = int(t)

    def save_to_csv(self):
        if self.data_log:
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'Error (m)', 'Avg Error (m)', 'v (m/s)', 'w (rad/s)'])
                writer.writerows(self.data_log)
            self.get_logger().info(f'Saved MPC data to {self.csv_filename}')

def main():
    rclpy.init()
    node = MPCLTVFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.save_to_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()