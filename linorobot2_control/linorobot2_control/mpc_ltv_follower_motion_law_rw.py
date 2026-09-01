import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import QoSProfile, DurabilityPolicy
import math
import csv
import datetime
import numpy as np

class RealRobotMPCFollower(Node):
    def __init__(self):
        # 1. DÙNG THỜI GIAN THỰC
        super().__init__(
            'mpc_follower',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)
            ]
        )

        # 2. PARAMETERS
        self.declare_parameter('offset_x', 1.1)
        self.declare_parameter('offset_y', 0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('h', 5)
        self.declare_parameter('num_loops', 10)

        self.off_x = self.get_parameter('offset_x').value
        self.off_y = self.get_parameter('offset_y').value
        self.off_phi = self.get_parameter('offset_phi').value
        self.Ts = self.get_parameter('Ts').value
        self.h = self.get_parameter('h').value
        self.num_loops = self.get_parameter('num_loops').value

        self.get_logger().info(f"REAL ROBOT: Ts={self.Ts}s. Offset: {self.off_x}, {self.off_y}, {self.off_phi}")

        # Giới hạn vật lý
        self.v_max, self.w_max = 0.33, 3.85
        self.dv_max, self.dw_max = 1.0, 4.0
        self.e_xy_max = 0.05 / 5
        self.e_phi_max = 0.1 / 2
        self.vr_max = self.v_max / 10
        self.wr_max = self.w_max / 10
        self.v_path_max = 0.8
        self.a_path_max = 0.5
        self.ar = 0.65

        # REAL ROBOT
        self.h = 6
        self.ar = 0.85
        self.v_max = 0.3
        self.w_max = 2.0

        self.e_xy_max = 0.05 / 5
        self.e_phi_max = 0.1 / 2
        self.vr_max = self.v_max / 15
        self.wr_max = self.w_max / 15

        self.dv_max = 0.5
        self.dw_max = 2.0
        self.v_path_max = 0.4
        self.a_path_max = 0.2

        # 3. TIỀN TÍNH TOÁN QUỸ ĐẠO
        self.get_logger().info("Đang tính toán quỹ đạo MPC cho robot thật...")
        self.s_end_one = 30.0
        self.s_end = self.s_end_one * self.num_loops
        self.A_size = 0.7
        self.freq = 2 * math.pi / self.s_end_one
        
        self.ref_q, self.ref_u = self.precompute_path(self.v_path_max, self.a_path_max)
        self.max_k = len(self.ref_q) - self.h - 1
        
        # 4. KHỞI TẠO TRỌNG SỐ MPC
        self.init_mpc_weights(self.e_xy_max, self.e_phi_max, self.vr_max, self.wr_max, self.ar)

        # 5. BIẾN TRẠNG THÁI
        self.k = 0
        self.curr_q = [self.off_x, self.off_y, self.off_phi]
        self.v_prev, self.w_prev = 0.0, 0.0
        self.start_time = None
        self.odom_received = False
        self.sum_e_dist_sq = 0.0
        self.count = 0
        self.last_log_t = -1
        self.data_log = []
        self.csv_filename = f'real_motion_law_mpc_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

        # 6. ROS INFRASTRUCTURE
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.actual_path_pub = self.create_publisher(Path, '/robot_path', 10)
        qos_p = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ref_path_pub = self.create_publisher(Path, '/ref_path', qos_p)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'
        
        # 7. TIMERS
        self.ref_timer = self.create_timer(5.0, self.publish_static_ref_path)
        self.timer = self.create_timer(self.Ts, self.control_loop)
        
        self.get_logger().info(f"MPC Follower READY.")

    def init_mpc_weights(self, e_xy_max, e_phi_max, vr_max, wr_max, ar):
        # Trọng số phạt sai số (Q) và phạt nỗ lực điều khiển (R)
        # Robot thật cần phạt R nặng hơn để lệnh v, w mượt mà
        e_xy_max = e_xy_max
        e_phi_max = e_phi_max
        vr_max = vr_max
        wr_max = wr_max
        
        q_diag = np.array([1/e_xy_max**2, 1/e_xy_max**2, 1/e_phi_max**2])
        r_diag = np.array([1/vr_max**2, 1/wr_max**2])
        
        self.Qt = np.kron(np.eye(self.h), np.diag(q_diag))
        self.Rt = np.kron(np.eye(self.h), np.diag(r_diag))
        
        ar = ar # Tốc độ hội tụ sai số
        self.Fr = np.zeros((3 * self.h, 3))
        for i in range(1, self.h + 1):
            self.Fr[(i-1)*3 : i*3, :] = np.eye(3) * (ar**i)

    def precompute_path(self, v_path_max, a_path_max):
        s_tmp, v_s = 0.0, 0.01
        v_path_max = v_path_max;
        a_path_max = a_path_max;
        s_list, v_list = [], []
        
        while s_tmp <= self.s_end:
            dist = self.s_end - s_tmp
            a_s = a_path_max if s_tmp < 1.0 else (-a_path_max if dist < 0.5 else 0.0)
            v_s = max(min(v_s + a_s * self.Ts, v_path_max), 0.02)
            s_tmp += v_s * self.Ts
            s_list.append(s_tmp); v_list.append(v_s)
        
        for _ in range(self.h + 10):
            s_list.append(s_list[-1]); v_list.append(0.0)
            
        q_ref, u_ref = [], []
        for i in range(len(s_list)):
            ss, vs = s_list[i], v_list[i]
            xr = 1.1 + self.A_size * math.sin(self.freq * ss)
            yr = 0.9 + self.A_size * math.sin(2 * self.freq * ss)
            dx_ds = self.A_size * self.freq * math.cos(self.freq * ss)
            dy_ds = 2 * self.A_size * self.freq * math.cos(2 * self.freq * ss)
            dxr, dyr = dx_ds * vs, dy_ds * vs
            
            # Tính đạo hàm bậc 2 cho w_ref
            d2x_ds2 = -self.A_size * (self.freq**2) * math.sin(self.freq * ss)
            d2y_ds2 = -4 * self.A_size * (self.freq**2) * math.sin(2 * self.freq * ss)
            # Giả định a_s=0 ở bước ref
            ddxr, ddyr = d2x_ds2*(vs**2), d2y_ds2*(vs**2)
            
            v_r = math.sqrt(dxr**2 + dyr**2)
            w_r = (dxr*ddyr - dyr*ddxr)/(dxr**2 + dyr**2 + 1e-6)
            
            q_ref.append([xr, yr, math.atan2(dyr, dxr)])
            u_ref.append([v_r, w_r])
            
        return np.array(q_ref), np.array(u_ref)

    def publish_static_ref_path(self):
        msg = Path()
        msg.header.frame_id, msg.header.stamp = 'world', self.get_clock().now().to_msg()
        for i in range(len(self.ref_q)//self.num_loops):
            p = PoseStamped()
            p.header.frame_id = 'world'
            p.pose.position.x, p.pose.position.y = self.ref_q[i][0], self.ref_q[i][1]
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
        self.odom_received = True

    def control_loop(self):
        if not self.odom_received or self.k > self.max_k: return
        now = self.get_clock().now()
        if self.start_time is None: self.start_time = now; return
        t = (now - self.start_time).nanoseconds / 1e9

        # A. MPC MATRICES (Tuyến tính hóa rời rạc)
        B = np.array([[self.Ts, 0], [0, 0], [0, self.Ts]])
        Hm, Fm = np.zeros((3 * self.h, 2 * self.h)), np.zeros((3 * self.h, 3))
        A_cum = np.eye(3)

        for i in range(1, self.h + 1):
            vr, wr = self.ref_u[self.k + i - 1]
            Ai = np.array([[1, self.Ts*wr, 0], [-self.Ts*wr, 1, self.Ts*vr], [0, 0, 1]])
            A_cum = Ai @ A_cum
            Fm[(i-1)*3 : i*3, :] = A_cum
            A_forced = np.eye(3)
            for j in range(i, 0, -1):
                Hm[(i-1)*3 : i*3, (j-1)*2 : j*2] = A_forced @ B
                vrj, wrj = self.ref_u[self.k + j - 1]
                Aj = np.array([[1, self.Ts*wrj, 0], [-self.Ts*wrj, 1, self.Ts*vrj], [0, 0, 1]])
                A_forced = Aj @ A_forced

        # B. OPTIMAL CONTROL LAW
        phi = self.curr_q[2]
        ex_g = self.ref_q[self.k][0] - self.curr_q[0]
        ey_g = self.ref_q[self.k][1] - self.curr_q[1]
        ephi_g = math.atan2(math.sin(self.ref_q[self.k][2] - phi), math.cos(self.ref_q[self.k][2] - phi))
        
        # Chuyển về Body Frame
        e = np.array([[math.cos(phi), math.sin(phi), 0], [-math.sin(phi), math.cos(phi), 0], [0, 0, 1]]) @ np.array([ex_g, ey_g, ephi_g])

        # Giải tối ưu: u = -(H'*Q*H + R)^-1 * H'*Q * (Fr - Fm)*e
        inv_term = np.linalg.inv(Hm.T @ self.Qt @ Hm + self.Rt)
        u_mpc = -(inv_term @ (Hm.T @ self.Qt @ (self.Fr - Fm))) @ e
        
        v_target = self.ref_u[self.k][0] * math.cos(e[2]) + u_mpc[0]
        w_target = self.ref_u[self.k][1] + u_mpc[1]

        # C. RATE LIMIT & SATURATION
        v_final = self.v_prev + np.clip(v_target - self.v_prev, -self.dv_max*self.Ts, self.dv_max*self.Ts)
        v_final = np.clip(v_final, -self.v_max, self.v_max)
        
        w_final = self.w_prev + np.clip(w_target - self.w_prev, -self.dw_max*self.Ts, self.dw_max*self.Ts)
        w_final = np.clip(w_final, -self.w_max, self.w_max)

        self.v_prev, self.w_prev = v_final, w_final
        self.k += 1

        # D. EXECUTE & LOG
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v_final), float(w_final)
        self.cmd_pub.publish(cmd)
        self.update_path_and_log(t, math.sqrt(ex_g**2 + ey_g**2), v_final, w_final)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info("=== MPC STOPPED ===")

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
            self.get_logger().info(f"Log: {self.csv_filename}")

def main():
    rclpy.init()
    node = RealRobotMPCFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.stop_robot()
        node.save_csv(); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': 
    main()