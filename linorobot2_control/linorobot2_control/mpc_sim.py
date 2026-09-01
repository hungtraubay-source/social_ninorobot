#!/usr/bin/env python3
"""
mpc_nav2_expert_follower_v3_FIXED.py
Bộ điều khiển bám quỹ đạo MPC - Đã sửa các vấn đề lý thuyết
(In-place Rotation, Aggressive Smoothing, Target Tethering, Numerical Stability)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException


# ===========================================================================
#  MPC CONTROLLER - FIXED VERSION
# ===========================================================================
class MPCController:
    def __init__(self, Ts=0.05, h=12, ar=0.12, 
                 v_max=0.25, w_max=1.5, dv_max=0.8, dw_max=4.0):
        self.Ts = Ts
        self.h = h
        self.ar = ar
        
        self.v_max = v_max
        self.w_max = w_max
        self.dv_max = dv_max
        self.dw_max = dw_max

        self.v_prev = 0.0
        self.w_prev = 0.0
        self.e3_prev = 0.0  # [FIX 4] Lưu sai số góc trước để tính D-term

        # [ĐỒNG BỘ HA] Set trọng số Bryson's rule theo đúng MAX_E của HA
        e_xy_max = 0.08         # Tương đương MAX_E1, MAX_E2 = 0.1
        e_phi_max = math.pi / 3 # Tương đương MAX_E3 = 45 độ
        vr_max = self.v_max / 1.5 
        wr_max = self.w_max / 1.5 

        q_diag = np.array([1/e_xy_max**2, 1/e_xy_max**2, 1/e_phi_max**2])
        r_diag = np.array([1/vr_max**2, 1/wr_max**2])
        
        self.Qt = np.kron(np.eye(self.h), np.diag(q_diag))
        self.Rt = np.kron(np.eye(self.h), np.diag(r_diag))
        
        self.Fr = np.zeros((3 * self.h, 3))
        for i in range(1, self.h + 1):
            self.Fr[(i-1)*3 : i*3, :] = np.eye(3) * (self.ar**i)

    def compute(self, rx, ry, rphi, traj, s_target, v_s, a_s):
        """
        Compute MPC control command.
        
        Args:
            rx, ry, rphi: Current robot state
            traj: Trajectory object
            s_target: Target arc-length parameter
            v_s: Current virtual velocity
            a_s: Current virtual acceleration [FIX 3]
        
        Returns:
            v_final, w_final, debug_dict
        """
        # --- TÍNH SAI SỐ HIỆN TẠI TRƯỚC ĐỂ KIỂM TRA ĐIỀU KIỆN ---
        x_ref0, y_ref0 = traj.pos(s_target)
        phi_ref0 = float(np.interp(s_target, traj.s_arr, traj.phis))
        
        ex_g = x_ref0 - rx
        ey_g = y_ref0 - ry
        ephi_g = math.atan2(math.sin(phi_ref0 - rphi), math.cos(phi_ref0 - rphi))
        
        # Chuyển đổi sang hệ tọa độ Robot (Kanayama Error)
        e = np.array([[math.cos(rphi), math.sin(rphi), 0], 
                      [-math.sin(rphi), math.cos(rphi), 0], 
                      [0, 0, 1]]) @ np.array([ex_g, ey_g, ephi_g])
        e1, e2, e3 = e[0], e[1], e[2]

        # [FIX 1] IN-PLACE ROTATION BYPASS (Chống mù góc hẹp)
        if abs(e3) > math.radians(45.0):
            v_target = 0.0
            
            # [FIX 4] PD Controller cho in-place rotation
            # P-gain = 2.0, D-gain = 0.1 (smooth damping)
            e3_dot = (e3 - self.e3_prev) / self.Ts
            w_target = 2.0 * e3 + 0.1 * e3_dot
            self.e3_prev = e3
            
            # Áp dụng giới hạn Rate Limit và Saturation cho In-place
            v_final = self.v_prev + np.clip(v_target - self.v_prev, -self.dv_max*self.Ts, self.dv_max*self.Ts)
            w_final = self.w_prev + np.clip(w_target - self.w_prev, -self.dw_max*self.Ts, self.dw_max*self.Ts)
            v_final = float(np.clip(v_final, -self.v_max, self.v_max))
            w_final = float(np.clip(w_final, -self.w_max, self.w_max))
            
            self.v_prev, self.w_prev = v_final, w_final
            return v_final, w_final, {'e1': e1, 'e2': e2, 'e3': e3, 'mode': 'IN-PLACE', 'v_r': 0.0, 'w_r': 0.0}

        # --- NẾU ĐỦ ĐIỀU KIỆN, CHẠY MPC ---
        # 1. Trích xuất Horizon với dự báo vận tốc [FIX 3]
        ref_horizon = []
        for i in range(self.h):
            s_i = s_target + i * (v_s * self.Ts)
            if s_i >= traj.L_total:
                x_r, y_r = traj.pos(traj.L_total)
                phi_r = traj.phis[-1]
                v_r, w_r = 0.0, 0.0
            else:
                x_r, y_r = traj.pos(s_i)
                phi_r = float(np.interp(s_i, traj.s_arr, traj.phis))
                kappa = float(np.interp(s_i, traj.s_arr, traj.kappas))
                
                # [FIX 3] Dự báo vận tốc tương lai thay v�� dùng v_s hiện tại
                v_r = max(v_s + a_s * i * self.Ts, 0.01)  # Không âm
                w_r = kappa * v_r
            
            ref_horizon.append((x_r, y_r, phi_r, v_r, w_r))

        # 2. Tuyến tính hóa và tạo ma trận Hm, Fm
        B = np.array([[self.Ts, 0], [0, 0], [0, self.Ts]])
        Hm = np.zeros((3 * self.h, 2 * self.h))
        Fm = np.zeros((3 * self.h, 3))
        A_cum = np.eye(3)

        for i in range(1, self.h + 1):
            xr, yr, phir, vr, wr = ref_horizon[i-1]
            Ai = np.array([[1, self.Ts*wr, 0], [-self.Ts*wr, 1, self.Ts*vr], [0, 0, 1]])
            
            A_cum = Ai @ A_cum
            Fm[(i-1)*3 : i*3, :] = A_cum
            
            A_forced = np.eye(3)
            for j in range(i, 0, -1):
                Hm[(i-1)*3 : i*3, (j-1)*2 : j*2] = A_forced @ B
                xr_j, yr_j, phir_j, vr_j, wr_j = ref_horizon[j-1]
                Aj = np.array([[1, self.Ts*wr_j, 0], [-self.Ts*wr_j, 1, self.Ts*vr_j], [0, 0, 1]])
                A_forced = Aj @ A_forced

        # 3. Giải bài toán LQR cho MPC [FIX 2] - Matrix Conditioning Check
        try:
            H_matrix = Hm.T @ self.Qt @ Hm + self.Rt
            
            # Check condition number
            cond_num = np.linalg.cond(H_matrix)
            if cond_num > 1e10:
                # Matrix ill-conditioned, dùng pseudo-inverse
                inv_term = np.linalg.pinv(H_matrix)
            else:
                inv_term = np.linalg.inv(H_matrix)
                
        except np.linalg.LinAlgError:
            # Matrix singular, fallback to pseudo-inverse
            inv_term = np.linalg.pinv(Hm.T @ self.Qt @ Hm + self.Rt)
        
        KKgpc = inv_term @ (Hm.T @ self.Qt @ (self.Fr - Fm))
        u_mpc = -KKgpc[0:2, :] @ e
        
        v_target = ref_horizon[0][3] * math.cos(e[2]) + u_mpc[0]
        w_target = ref_horizon[0][4] + u_mpc[1]

        # 4. Rate Limit & Saturation
        v_final = self.v_prev + np.clip(v_target - self.v_prev, -self.dv_max*self.Ts, self.dv_max*self.Ts)
        v_final = float(np.clip(v_final, -self.v_max, self.v_max))
        
        w_final = self.w_prev + np.clip(w_target - self.w_prev, -self.dw_max*self.Ts, self.dw_max*self.Ts)
        w_final = float(np.clip(w_final, -self.w_max, self.w_max))

        self.v_prev, self.w_prev = v_final, w_final
        self.e3_prev = e3
        
        return v_final, w_final, {
            'e1': e1, 'e2': e2, 'e3': e3, 'mode': 'MPC',
            'v_r': ref_horizon[0][3], 'w_r': ref_horizon[0][4],
            'v_pred_error': v_target - v_final
        }

    def reset(self):
        self.v_prev = self.w_prev = 0.0
        self.e3_prev = 0.0


# ===========================================================================
#  NAV2 TRAJECTORY
# ===========================================================================
class Nav2Trajectory:
    def __init__(self, path_msg: Path):
        self.xs = np.array([p.pose.position.x for p in path_msg.poses])
        self.ys = np.array([p.pose.position.y for p in path_msg.poses])

        if len(self.xs) < 2:
            self.s_arr   = np.array([0.0])
            self.L_total = 0.0
            return

        dx = np.diff(self.xs)
        dy = np.diff(self.ys)
        ds = np.hypot(dx, dy)
        self.s_arr   = np.concatenate([[0.0], np.cumsum(ds)])
        self.L_total = self.s_arr[-1]

        self.phis = np.zeros_like(self.xs)
        for i in range(len(self.xs) - 1):
            self.phis[i] = math.atan2(self.ys[i+1] - self.ys[i], self.xs[i+1] - self.xs[i])
        self.phis[-1] = self.phis[-2]
        self.phis = np.unwrap(self.phis)

        self.kappas = np.zeros_like(self.xs)
        for i in range(1, len(self.xs) - 1):
            ds_val = self.s_arr[i+1] - self.s_arr[i-1]
            if ds_val > 1e-4:
                self.kappas[i] = (self.phis[i+1] - self.phis[i-1]) / ds_val
        k = min(7, len(self.kappas))
        self.kappas = np.convolve(self.kappas, np.ones(k) / k, mode='same')

    def pos(self, s):
        s = np.clip(s, 0.0, self.L_total)
        return (float(np.interp(s, self.s_arr, self.xs)),
                float(np.interp(s, self.s_arr, self.ys)))

    def find_nearest_s(self, rx, ry):
        if self.L_total == 0.0: return 0.0
        return float(self.s_arr[np.argmin(np.hypot(self.xs - rx, self.ys - ry))])

    def get_cross_track_error(self, rx, ry):
        if self.L_total == 0.0: return 0.0, 0.0
        nearest_idx = np.argmin(np.hypot(self.xs - rx, self.ys - ry))
        nearest_s = self.s_arr[nearest_idx]
        x_n, y_n = self.pos(nearest_s)
        cte = math.hypot(rx - x_n, ry - y_n)
        return cte, float(nearest_s)

    def curvature(self, s):
        s = np.clip(s, 0.0, self.L_total)
        return abs(float(np.interp(s, self.s_arr, self.kappas)))


# ===========================================================================
#  MAIN ROS2 NODE
# ===========================================================================
class MPCNav2ExpertNode(Node):

    LA_K    = 2.5    
    LA_MIN  = 0.20   
    LA_MAX  = 0.50 

    def __init__(self):
        super().__init__(
            'mpc_nav2_expert',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)
            ]
        )
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('v_path_max', 0.25)   
        self.declare_parameter('a_path_max', 0.20)

        self.Ts         = self.get_parameter('Ts').value
        self.v_path_max = self.get_parameter('v_path_max').value
        self.a_path_max = self.get_parameter('a_path_max').value

        self.curve_beta     = 0.10
        self.num_curve_samp = 10
        self.stop_zone      = 0.08   

        self.get_logger().info(f'MPC Expert (HA Synced - FIXED) | Ts={self.Ts}s | v_path_max={self.v_path_max}m/s')

        self.traj = None
        self.ctrl = MPCController(
            Ts=self.Ts, h=10, ar=0.15,
            v_max=self.v_path_max, w_max=1.5, 
            dv_max=0.8, dw_max=4.0
        )

        self.s             = 0.0
        self.v_s           = 0.01
        self.a_s           = 0.0  # [FIX 3] Lưu acceleration hiện tại
        self.curr_q        = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.path_received = False
        self.s_initialized = False

        self.sum_cross_sq = 0.0
        self.log_count    = 0
        self.max_cross    = 0.0

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.timer    = self.create_timer(self.Ts, self.control_loop)

        self.get_logger().info('Node MPC Nav2 Follower (FIXED) san sang.')

    def _look_ahead(self):
        return float(np.clip(self.LA_K * self.v_s, self.LA_MIN, self.LA_MAX))

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2: return
        self.traj = Nav2Trajectory(msg)
        self.path_received = True

        if self.pose_received:
            self.s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
            self.s_initialized = True
        else:
            self.s = 0.0
            self.s_initialized = False

        self.v_s = 0.01
        self.a_s = 0.0
        self.ctrl.reset()
        self.sum_cross_sq = 0.0
        self.max_cross    = 0.0
        self.log_count    = 0
        self.get_logger().info(f'=== QUY DAO MOI: {self.traj.L_total:.2f}m ===')

    def _v_safe(self, s):
        if self.traj is None: return self.v_path_max
        la = self._look_ahead()
        n = max(self.num_curve_samp, 1)
        vmin = self.v_path_max
        for i in range(n + 1):
            kap = self.traj.curvature(s + i / n * la)
            v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
            if v_ok < vmin: vmin = v_ok
        return max(vmin, 0.02)

    def control_loop(self):
        try:
            tf_trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            rx = tf_trans.transform.translation.x
            ry = tf_trans.transform.translation.y
            q = tf_trans.transform.rotation
            rphi = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y**2 + q.z**2))
            self.curr_q = [rx, ry, rphi]
            self.pose_received = True
        except TransformException:
            return

        if not self.path_received or self.traj is None: return

        rx, ry, rphi = self.curr_q
        if not self.s_initialized:
            self.s = self.traj.find_nearest_s(rx, ry)
            self.s_initialized = True

        # Tính toán khoảng cách và lỗi Cross-track
        cte, nearest_s = self.traj.get_cross_track_error(rx, ry)
        v_adapt = min(self._v_safe(self.s), self.ctrl.v_max)
        dist_left = self.traj.L_total - self.s
        dist_brake = self.v_s**2 / (2.0 * self.a_path_max + 1e-9)

        # Logic gia tốc mặc định
        if dist_left <= dist_brake:
            self.a_s = -self.a_path_max
        elif self.v_s < v_adapt - 0.005:
            self.a_s =  self.a_path_max
        elif self.v_s > v_adapt + 0.005:
            self.a_s = -self.a_path_max
        else:
            self.a_s = 0.0

        # [FIX 1] TARGET TETHERING - Sửa dấu along_track_err
        # along_track_err > 0 nghĩa là xe TỤT LẠI phía sau mục tiêu ảo
        along_track_err = nearest_s - self.s  # FIXED: Đổi dấu!
        if along_track_err > 0.25 or cte > 0.25:
            self.a_s = -self.a_path_max * 0.5

        # Cập nhật vận tốc ảo và vị trí ảo
        self.v_s = float(np.clip(self.v_s + self.a_s * self.Ts, 0.01, self.ctrl.v_max))
        self.s  += self.v_s * self.Ts
        
        # Clamp look-ahead
        la = self._look_ahead()
        if self.s > nearest_s + la:
            self.s = nearest_s + la
            
        self.s = min(self.s, self.traj.L_total)

        # Kiểm tra đích
        x_end, y_end = self.traj.xs[-1], self.traj.ys[-1]
        dist_to_goal = math.hypot(rx - x_end, ry - y_end)

        if dist_to_goal <= self.stop_zone and (self.s > self.traj.L_total * 0.9):
            self._stop_robot()
            rms = math.sqrt(self.sum_cross_sq / max(self.log_count, 1))
            self.get_logger().info(f'=== HOAN THANH TAI DICH: RMS={rms:.4f}m ===')
            self.path_received = False
            return

        # Tính toán MPC [FIX 3] - Truyền a_s vào compute()
        v_cmd, w_cmd, dbg = self.ctrl.compute(rx, ry, rphi, self.traj, self.s, self.v_s, self.a_s)

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_cmd, w_cmd
        self.cmd_pub.publish(cmd)

        # Cập nhật Log
        self.sum_cross_sq += cte**2
        self.log_count += 1
        if cte > self.max_cross: self.max_cross = cte

        if self.log_count % 10 == 0:
            rms_cross = math.sqrt(self.sum_cross_sq / self.log_count)
            # [FIX 5] Log thêm debug info: reference velocity, acceleration, along-track error
            self.get_logger().info(
                f"[{dbg['mode']}] v={v_cmd:.3f}/{dbg['v_r']:.3f}m/s w={w_cmd:+.3f}/{dbg['w_r']:+.3f} | "
                f"CTE={cte:.3f} ATE={along_track_err:+.3f} | "
                f"e1={dbg['e1']:+.3f} e2={dbg['e2']:+.3f} e3={math.degrees(dbg['e3']):+05.1f}° | "
                f"RMS={rms_cross:.4f} max={self.max_cross:.4f}"
            )

    def _stop_robot(self):
        stop = Twist()
        self.cmd_pub.publish(stop)
        for _ in range(3): self.cmd_pub.publish(stop)

def main(args=None):
    rclpy.init(args=args)
    node = MPCNav2ExpertNode() 
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()