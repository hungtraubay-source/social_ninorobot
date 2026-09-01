#!/usr/bin/env python3
"""
ha_nav2_expert_follower_v2.py
Bộ điều khiển bám quỹ đạo HA 12 Terms - Đã tinh chỉnh giám sát sai số
- Hiển thị cross-track error và along-track error thực tế.
- Đồng bộ khởi tạo s giữa pose và path.
- Log text thuần, không màu mè, tối ưu cho Terminator.
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
#  HEDGE ALGEBRA (12 ngữ nghĩa) - GIỮ NGUYÊN
# ===========================================================================
class HedgeAlgebra:
    def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
        self.theta = theta
        self.alpha = alpha
        self.beta  = beta
        self.fm    = {'V': alpha, 'L': beta, 'Small': theta, 'Large': 1.0 - theta}

        self.term_specs = [
            ('Zero',     [],          None   ),
            ('VVSmall',  ['V', 'V'],  'Small'),
            ('VSmall',   ['V'],       'Small'),
            ('Small',    [],          'Small'),
            ('VLSmall',  ['V', 'L'],  'Small'),
            ('LSmall',   ['L'],       'Small'),
            ('Neutral',  [],          None   ),
            ('LLarge',   ['L'],       'Large'),
            ('Large',    [],          'Large'),
            ('VLarge',   ['V'],       'Large'),
            ('VVLarge',  ['V', 'V'],  'Large'),
            ('Absolute', [],          None   ),
        ]

        self.sem = {n: self._compute_sem(n, h, b) for n, h, b in self.term_specs}
        self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])

        self.hcnt = {
            name: len(hedges) if name not in {'Zero', 'Neutral', 'Absolute', 'Small', 'Large'} else 0
            for name, hedges, _ in self.term_specs
        }

    def _sgn_hedge(self, h): return 1 if h == 'V' else -1
    def _sgn_base(self, h, base): return -1 if h == 'V' else 1 if base == 'Small' else 1 if h == 'V' else -1

    def _compute_sem(self, name, hedges, base):
        if name == 'Zero':     return 0.0
        if name == 'Neutral':  return self.theta
        if name == 'Absolute': return 1.0
        fm_base   = self.fm[base]
        current_v = (self.theta - self.alpha * fm_base if base == 'Small' else self.theta + self.alpha * fm_base)
        if not hedges: return current_v
        current_fm = fm_base
        for h in reversed(hedges):
            fm_hx  = (self.beta if h == 'V' else self.alpha) * current_fm
            sum_fm = fm_hx if h == 'V' else current_fm
            current_v += self._sgn_base(h, base) * (sum_fm - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha)) * fm_hx)
            current_fm = fm_hx
        return current_v

    def fuzzify(self, value):
        v     = float(np.clip(value, 0.0, 1.0))
        min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
        cands = [t for t in self.ordered_terms if abs(self.sem[t] - v) <= min_d + 1e-9]
        return min(cands, key=lambda t: self.hcnt.get(t, 3))

    def infer(self, input_val, rule_base):
        term = self.fuzzify(float(np.clip(input_val, 0.0, 1.0)))
        return self.sem[rule_base.get(term, 'Neutral')]


# ===========================================================================
#  RULE BASES - GIỮ NGUYÊN
# ===========================================================================
def _rule_longitudinal():
    return {'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero', 'Small': 'VSmall', 'VLSmall': 'Small', 
            'LSmall': 'VLSmall', 'Neutral': 'LSmall', 'LLarge': 'Large', 'Large': 'Large', 
            'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}

def _rule_lateral():
    return {'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero', 'Small': 'VSmall', 'VLSmall': 'Small', 
            'LSmall': 'LSmall', 'Neutral': 'Large', 'LLarge': 'VLarge', 'Large': 'VLarge', 
            'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}

def _rule_heading():
    return {'Zero': 'Zero', 'VVSmall': 'VVSmall', 'VSmall': 'VSmall', 'Small': 'Small', 'VLSmall': 'VLSmall', 
            'LSmall': 'LSmall', 'Neutral': 'Large', 'LLarge': 'LLarge', 'Large': 'VLarge', 
            'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}


# ===========================================================================
#  HA CONTROLLER - GIỮ NGUYÊN
# ===========================================================================
class HAController:
    MAX_E1 = 10.0
    MAX_E2 = 10.0
    MAX_E3 = math.pi / 3

    def __init__(self, ha, gain_e1=1.0, gain_e2=2.0, gain_e3=1.5,
                 v_max=0.05, w_max=6.5, dv_max=1.5, dw_max=3.0, Ts=0.05, lpf_alpha=0.5):
        self.ha         = ha
        self.gain_e1    = gain_e1
        self.gain_e2    = gain_e2
        self.gain_e3    = gain_e3
        self.v_max      = v_max
        self.w_max      = w_max
        self.dv_max     = dv_max
        self.dw_max     = dw_max
        self.Ts         = Ts
        self.lpf_alpha  = lpf_alpha
        self.rule_long  = _rule_longitudinal()
        self.rule_lat   = _rule_lateral()
        self.rule_head  = _rule_heading()
        self.v_prev     = 0.0
        self.w_prev     = 0.0
        self.w_lpf_prev = 0.0

    def _ha_fb(self, err, max_e, rule, gain):
        if abs(err) <= 0.01 * max_e: return 0.0
        norm = min(abs(err) / max_e, 1.0)
        mag  = self.ha.infer(norm, rule) * gain
        return math.copysign(mag, err)

    def compute(self, v_ref, w_ref, e1, e2, e3):
        fb1 = self._ha_fb(e1, self.MAX_E1, self.rule_long, self.gain_e1)
        fb2 = self._ha_fb(e2, self.MAX_E2, self.rule_lat,  self.gain_e2)
        fb3 = self._ha_fb(e3, self.MAX_E3, self.rule_head, self.gain_e3)

        v_raw = v_ref * math.cos(e3) + fb1
        w_raw = w_ref + fb2 + fb3

        w_filt = self.lpf_alpha * self.w_lpf_prev + (1 - self.lpf_alpha) * w_raw
        self.w_lpf_prev = w_filt

        v_sat = float(np.clip(v_raw,  -self.v_max, self.v_max))
        w_sat = float(np.clip(w_filt, -self.w_max, self.w_max))

        dt  = self.Ts
        dv  = float(np.clip(v_sat - self.v_prev, -self.dv_max * dt, self.dv_max * dt))
        dw  = float(np.clip(w_sat - self.w_prev, -self.dw_max * dt, self.dw_max * dt))

        v_cmd = self.v_prev + dv
        w_cmd = self.w_prev + dw
        self.v_prev = v_cmd
        self.w_prev = w_cmd
        return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2, 'fb3': fb3}

    def reset(self):
        self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


# ===========================================================================
#  NAV2 TRAJECTORY - BỔ SUNG PHƯƠNG THỨC TÍNH CROSS-TRACK
# ===========================================================================
class Nav2Trajectory:
    def __init__(self, path_msg: Path):
        self.xs = np.array([p.pose.position.x for p in path_msg.poses])
        self.ys = np.array([p.pose.position.y for p in path_msg.poses])
        
        if len(self.xs) < 2:
            self.s_arr = np.array([0.0])
            self.L_total = 0.0
            return

        dx = np.diff(self.xs)
        dy = np.diff(self.ys)
        ds = np.hypot(dx, dy)
        self.s_arr = np.concatenate([[0.0], np.cumsum(ds)])
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

    def pos(self, s):
        s = np.clip(s, 0.0, self.L_total)
        x = float(np.interp(s, self.s_arr, self.xs))
        y = float(np.interp(s, self.s_arr, self.ys))
        return x, y

    def curvature(self, s):
        s = np.clip(s, 0.0, self.L_total)
        return abs(float(np.interp(s, self.s_arr, self.kappas)))

    def ref_kinematics(self, s, v_s, a_s):
        s = np.clip(s, 0.0, self.L_total)
        phi_ref = float(np.interp(s, self.s_arr, self.phis))
        kappa = float(np.interp(s, self.s_arr, self.kappas))
        v_ref = v_s
        w_ref = kappa * v_s
        return phi_ref, v_ref, w_ref

    def find_nearest_s(self, rx, ry):
        if self.L_total == 0.0: return 0.0
        dists = np.hypot(self.xs - rx, self.ys - ry)
        return float(self.s_arr[np.argmin(dists)])

    def get_cross_track_error(self, rx, ry):
        """
        Tính khoảng cách vuông góc từ (rx, ry) đến điểm gần nhất trên path.
        Trả về: cross_track_error (luôn dương), nearest_s
        """
        if self.L_total == 0.0:
            return 0.0, 0.0
        nearest_idx = np.argmin(np.hypot(self.xs - rx, self.ys - ry))
        nearest_s = self.s_arr[nearest_idx]
        x_near, y_near = self.pos(nearest_s)
        cross_error = math.hypot(rx - x_near, ry - y_near)
        return cross_error, nearest_s


# ===========================================================================
#  MAIN ROS2 NODE - ĐÃ SỬA LOGIC KHỞI TẠO & GIÁM SÁT
# ===========================================================================
class HANav2ExpertNode(Node):

    def __init__(self):
        super().__init__('ha_nav2_expert')

        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('v_path_max', 0.4) # doi tu 0.4
        self.declare_parameter('a_path_max', 0.2)

        self.Ts         = self.get_parameter('Ts').value
        self.v_path_max = self.get_parameter('v_path_max').value
        self.a_path_max = self.get_parameter('a_path_max').value

        self.look_ahead_dist = 0.25 # 0.3
        self.curve_beta      = 0.15 # 0.06
        self.num_curve_samp  = 8

        self.get_logger().info(f'HA 12 Terms | Ts={self.Ts}s | v_max={self.v_path_max}m/s')

        self.traj = None
        self.ha   = HedgeAlgebra()
        self.ctrl = HAController(
            ha=self.ha,
            gain_e1=0.8, gain_e2=1.20, gain_e3=1.20,
            v_max=0.15,    w_max=2.0,
            dv_max=2.00,  dw_max=10.00,
            Ts=self.Ts,   lpf_alpha=0.65)

        self.s               = 0.0  
        self.v_s             = 0.02
        self.curr_q          = [0.0, 0.0, 0.0]
        self.pose_received   = False
        self.path_received   = False
        self.s_initialized   = False   # Cờ báo đã đồng bộ s với pose hiện tại

        # Biến giám sát chất lượng bám
        self.sum_cross_sq    = 0.0
        self.log_count       = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)

        self.timer = self.create_timer(self.Ts, self.control_loop)

        self.get_logger().info('Node HA Nav2 Follower da san sang (v2).')

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            return
        self.traj = Nav2Trajectory(msg)
        self.path_received = True

        # Nếu đã có pose, khởi tạo s ngay
        if self.pose_received:
            self.s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
            self.s_initialized = True
            self.get_logger().info(f'Khoi tao s = {self.s:.2f} m (pose da co)')
        else:
            self.s = 0.0
            self.s_initialized = False
            self.get_logger().info('Path nhan truoc pose, se khoi tao s khi co pose.')

        self.v_s = 0.02
        self.ctrl.reset()
        
        # Reset giám sát
        self.sum_cross_sq = 0.0
        self.log_count = 0
        
        self.get_logger().info(f'=== NHAN QUY DAO MOI: {self.traj.L_total:.2f} m ===')

    def _v_safe(self, s):
        if self.traj is None:
            return self.v_path_max
        v_min = self.v_path_max
        n = max(self.num_curve_samp, 1)
        for i in range(n + 1):
            kap  = self.traj.curvature(s + i / n * self.look_ahead_dist)
            v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
            if v_ok < v_min:
                v_min = v_ok
        return max(v_min, 0.03)

    def control_loop(self):
        # --- Cập nhật pose từ TF ---
        try:
            now = rclpy.time.Time()
            tf_trans = self.tf_buffer.lookup_transform('map', 'base_link', now)
            rx = tf_trans.transform.translation.x
            ry = tf_trans.transform.translation.y
            q  = tf_trans.transform.rotation
            rphi = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
            self.curr_q = [rx, ry, rphi]
            self.pose_received = True
        except TransformException:
            return

        # --- Kiểm tra điều kiện chạy ---
        if not self.pose_received or not self.path_received or self.traj is None:
            return

        # --- Khởi tạo trễ nếu path đến trước pose ---
        if not self.s_initialized:
            self.s = self.traj.find_nearest_s(rx, ry)
            self.s_initialized = True
            self.get_logger().info(f'Khoi tao s = {self.s:.2f} m (pose vua nhan)')

        # --- Tính vận tốc thích nghi và cập nhật s ---
        v_adapt    = self._v_safe(self.s)
        dist_brake = self.v_s**2 / (2 * self.a_path_max + 1e-9)
        dist_left  = self.traj.L_total - self.s

        if dist_left <= dist_brake:
            a_s = -self.a_path_max
        elif self.v_s < v_adapt - 0.01:
            a_s =  self.a_path_max
        elif self.v_s > v_adapt + 0.01:
            a_s = -self.a_path_max
        else:
            a_s =  0.0

        self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.v_path_max))
        self.s  += self.v_s * self.Ts

        # --- Điều chỉnh s không vượt quá xa robot ---
        cross_error, nearest_s = self.traj.get_cross_track_error(rx, ry)
        if self.s > nearest_s + self.look_ahead_dist:
            self.s = nearest_s + self.look_ahead_dist

        # --- Kiểm tra kết thúc quỹ đạo ---
        if self.s >= self.traj.L_total - 0.05:
            self.stop_robot()
            self.get_logger().info('=== HOAN THANH QUY DAO NAV2 ===')
            self.path_received = False
            return

        # --- Lấy tham chiếu tại điểm look-ahead ---
        phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
        x_ref, y_ref = self.traj.pos(self.s)

        # --- Tính sai số điều khiển e1, e2, e3 (dùng cho bộ điều khiển) ---
        rx, ry, rphi = self.curr_q
        dx_g = x_ref - rx
        dy_g = y_ref - ry
        cq = math.cos(rphi)
        sq = math.sin(rphi)

        e1 =  cq * dx_g + sq * dy_g
        e2 = -sq * dx_g + cq * dy_g
        e3 = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))

        # --- Tính toán lệnh vận tốc ---
        v_cmd, w_cmd, _ = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

        cmd = Twist()
        cmd.linear.x  = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

        # --- Giám sát chất lượng bám: cross-track error và along-track error ---
        cross_error, nearest_s = self.traj.get_cross_track_error(rx, ry)
        along_error = nearest_s - self.s   # >0: robot sau điểm look-ahead, <0: robot trước

        self.sum_cross_sq += cross_error**2
        self.log_count += 1
        rms_cross = math.sqrt(self.sum_cross_sq / self.log_count)

        # In log mỗi 10 vòng (2 Hz với Ts=0.05)
        if self.log_count % 10 == 0:
            msg = (
                f"v_cmd: {v_cmd:.3f} | w_cmd: {w_cmd:+.3f} || "
                f"cross: {cross_error:.3f} | along: {along_error:+.3f} || "
                f"e1: {e1:+.3f} | e2: {e2:+.3f} | e3: {math.degrees(e3):+05.1f} || "
                f"RMS_cross: {rms_cross:.3f}"
            )
            self.get_logger().info(msg)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = HANav2ExpertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C nhan duoc, dung robot.')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# """
# ha_nav2_expert_follower.py
# Bộ điều khiển bám quỹ đạo HA 12 Terms - Chuyên gia điều khiển
# Hiển thị log text thuần, không màu mè, tối ưu cho Terminator.
# """

# import math
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Path
# from tf2_ros.buffer import Buffer
# from tf2_ros.transform_listener import TransformListener
# from tf2_ros import TransformException


# # ===========================================================================
# #  HEDGE ALGEBRA (12 ngữ nghĩa)
# # ===========================================================================
# class HedgeAlgebra:
#     def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
#         self.theta = theta
#         self.alpha = alpha
#         self.beta  = beta
#         self.fm    = {'V': alpha, 'L': beta, 'Small': theta, 'Large': 1.0 - theta}

#         self.term_specs = [
#             ('Zero',     [],          None   ),
#             ('VVSmall',  ['V', 'V'],  'Small'),
#             ('VSmall',   ['V'],       'Small'),
#             ('Small',    [],          'Small'),
#             ('VLSmall',  ['V', 'L'],  'Small'),
#             ('LSmall',   ['L'],       'Small'),
#             ('Neutral',  [],          None   ),
#             ('LLarge',   ['L'],       'Large'),
#             ('Large',    [],          'Large'),
#             ('VLarge',   ['V'],       'Large'),
#             ('VVLarge',  ['V', 'V'],  'Large'),
#             ('Absolute', [],          None   ),
#         ]

#         self.sem = {n: self._compute_sem(n, h, b) for n, h, b in self.term_specs}
#         self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])

#         self.hcnt = {
#             name: len(hedges) if name not in {'Zero', 'Neutral', 'Absolute', 'Small', 'Large'} else 0
#             for name, hedges, _ in self.term_specs
#         }

#     def _sgn_hedge(self, h): return 1 if h == 'V' else -1
#     def _sgn_base(self, h, base): return -1 if h == 'V' else 1 if base == 'Small' else 1 if h == 'V' else -1

#     def _compute_sem(self, name, hedges, base):
#         if name == 'Zero':     return 0.0
#         if name == 'Neutral':  return self.theta
#         if name == 'Absolute': return 1.0
#         fm_base   = self.fm[base]
#         current_v = (self.theta - self.alpha * fm_base if base == 'Small' else self.theta + self.alpha * fm_base)
#         if not hedges: return current_v
#         current_fm = fm_base
#         for h in reversed(hedges):
#             fm_hx  = (self.beta if h == 'V' else self.alpha) * current_fm
#             sum_fm = fm_hx if h == 'V' else current_fm
#             current_v += self._sgn_base(h, base) * (sum_fm - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha)) * fm_hx)
#             current_fm = fm_hx
#         return current_v

#     def fuzzify(self, value):
#         v     = float(np.clip(value, 0.0, 1.0))
#         min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
#         cands = [t for t in self.ordered_terms if abs(self.sem[t] - v) <= min_d + 1e-9]
#         return min(cands, key=lambda t: self.hcnt.get(t, 3))

#     def infer(self, input_val, rule_base):
#         term = self.fuzzify(float(np.clip(input_val, 0.0, 1.0)))
#         return self.sem[rule_base.get(term, 'Neutral')]


# # ===========================================================================
# #  RULE BASES
# # ===========================================================================
# def _rule_longitudinal():
#     return {'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero', 'Small': 'VSmall', 'VLSmall': 'Small', 
#             'LSmall': 'VLSmall', 'Neutral': 'LSmall', 'LLarge': 'Large', 'Large': 'Large', 
#             'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}

# def _rule_lateral():
#     return {'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero', 'Small': 'VSmall', 'VLSmall': 'Small', 
#             'LSmall': 'LSmall', 'Neutral': 'Large', 'LLarge': 'VLarge', 'Large': 'VLarge', 
#             'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}

# def _rule_heading():
#     return {'Zero': 'Zero', 'VVSmall': 'VVSmall', 'VSmall': 'VSmall', 'Small': 'Small', 'VLSmall': 'VLSmall', 
#             'LSmall': 'LSmall', 'Neutral': 'Large', 'LLarge': 'LLarge', 'Large': 'VLarge', 
#             'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}


# # ===========================================================================
# #  HA CONTROLLER 
# # ===========================================================================
# class HAController:
#     MAX_E1 = 1.0
#     MAX_E2 = 1.0
#     MAX_E3 = math.pi / 5

#     def __init__(self, ha, gain_e1=0.5, gain_e2=1.5, gain_e3=2.5,
#                  v_max=0.15, w_max=3.85, dv_max=1.5, dw_max=4.0, Ts=0.05, lpf_alpha=0.65):
#         self.ha         = ha
#         self.gain_e1    = gain_e1
#         self.gain_e2    = gain_e2
#         self.gain_e3    = gain_e3
#         self.v_max      = v_max
#         self.w_max      = w_max
#         self.dv_max     = dv_max
#         self.dw_max     = dw_max
#         self.Ts         = Ts
#         self.lpf_alpha  = lpf_alpha
#         self.rule_long  = _rule_longitudinal()
#         self.rule_lat   = _rule_lateral()
#         self.rule_head  = _rule_heading()
#         self.v_prev     = 0.0
#         self.w_prev     = 0.0
#         self.w_lpf_prev = 0.0

#     def _ha_fb(self, err, max_e, rule, gain):
#         if abs(err) <= 0.01 * max_e: return 0.0
#         norm = min(abs(err) / max_e, 1.0)
#         mag  = self.ha.infer(norm, rule) * gain
#         return math.copysign(mag, err)

#     def compute(self, v_ref, w_ref, e1, e2, e3):
#         fb1 = self._ha_fb(e1, self.MAX_E1, self.rule_long, self.gain_e1)
#         fb2 = self._ha_fb(e2, self.MAX_E2, self.rule_lat,  self.gain_e2)
#         fb3 = self._ha_fb(e3, self.MAX_E3, self.rule_head, self.gain_e3)

#         v_raw = v_ref * math.cos(e3) + fb1
#         w_raw = w_ref + fb2 + fb3

#         w_filt = self.lpf_alpha * self.w_lpf_prev + (1 - self.lpf_alpha) * w_raw
#         self.w_lpf_prev = w_filt

#         v_sat = float(np.clip(v_raw,  -self.v_max, self.v_max))
#         w_sat = float(np.clip(w_filt, -self.w_max, self.w_max))

#         dt  = self.Ts
#         dv  = float(np.clip(v_sat - self.v_prev, -self.dv_max * dt, self.dv_max * dt))
#         dw  = float(np.clip(w_sat - self.w_prev, -self.dw_max * dt, self.dw_max * dt))

#         v_cmd = self.v_prev + dv
#         w_cmd = self.w_prev + dw
#         self.v_prev = v_cmd
#         self.w_prev = w_cmd
#         return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2, 'fb3': fb3}

#     def reset(self):
#         self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


# # ===========================================================================
# #  NAV2 TRAJECTORY
# # ===========================================================================
# class Nav2Trajectory:
#     def __init__(self, path_msg: Path):
#         self.xs = np.array([p.pose.position.x for p in path_msg.poses])
#         self.ys = np.array([p.pose.position.y for p in path_msg.poses])
        
#         if len(self.xs) < 2:
#             self.s_arr = np.array([0.0])
#             self.L_total = 0.0
#             return

#         dx = np.diff(self.xs)
#         dy = np.diff(self.ys)
#         ds = np.hypot(dx, dy)
#         self.s_arr = np.concatenate([[0.0], np.cumsum(ds)])
#         self.L_total = self.s_arr[-1]

#         self.phis = np.zeros_like(self.xs)
#         for i in range(len(self.xs) - 1):
#             self.phis[i] = math.atan2(self.ys[i+1] - self.ys[i], self.xs[i+1] - self.xs[i])
#         self.phis[-1] = self.phis[-2]
#         self.phis = np.unwrap(self.phis)

#         self.kappas = np.zeros_like(self.xs)
#         for i in range(1, len(self.xs) - 1):
#             ds_val = self.s_arr[i+1] - self.s_arr[i-1]
#             if ds_val > 1e-4:
#                 self.kappas[i] = (self.phis[i+1] - self.phis[i-1]) / ds_val

#     def pos(self, s):
#         s = np.clip(s, 0.0, self.L_total)
#         x = float(np.interp(s, self.s_arr, self.xs))
#         y = float(np.interp(s, self.s_arr, self.ys))
#         return x, y

#     def curvature(self, s):
#         s = np.clip(s, 0.0, self.L_total)
#         return abs(float(np.interp(s, self.s_arr, self.kappas)))

#     def ref_kinematics(self, s, v_s, a_s):
#         s = np.clip(s, 0.0, self.L_total)
#         phi_ref = float(np.interp(s, self.s_arr, self.phis))
#         kappa = float(np.interp(s, self.s_arr, self.kappas))
#         v_ref = v_s
#         w_ref = kappa * v_s
#         return phi_ref, v_ref, w_ref

#     def find_nearest_s(self, rx, ry):
#         if self.L_total == 0.0: return 0.0
#         dists = np.hypot(self.xs - rx, self.ys - ry)
#         return float(self.s_arr[np.argmin(dists)])


# # ===========================================================================
# #  MAIN ROS2 NODE 
# # ===========================================================================
# class HANav2ExpertNode(Node):

#     def __init__(self):
#         super().__init__('ha_nav2_expert')

#         self.declare_parameter('Ts', 0.05)
#         self.declare_parameter('v_path_max', 0.4)
#         self.declare_parameter('a_path_max', 0.2)

#         self.Ts         = self.get_parameter('Ts').value
#         self.v_path_max = self.get_parameter('v_path_max').value
#         self.a_path_max = self.get_parameter('a_path_max').value

#         self.look_ahead_dist = 0.30
#         self.curve_beta      = 0.06
#         self.num_curve_samp  = 8

#         self.get_logger().info(f'HA 12 Terms | Ts={self.Ts}s | v_max={self.v_path_max}m/s')

#         self.traj = None
#         self.ha   = HedgeAlgebra()
#         self.ctrl = HAController(
#             ha=self.ha,
#             gain_e1=0.8, gain_e2=1.20, gain_e3=1.20,
#             v_max=0.15,    w_max=2.0,
#             dv_max=2.00,  dw_max=10.00,
#             Ts=self.Ts,   lpf_alpha=0.65)

#         self.s             = 0.0  
#         self.v_s           = 0.02
#         self.curr_q        = [0.0, 0.0, 0.0]
#         self.pose_received = False
#         self.path_received = False

#         # Biến tính RMSE
#         self.sum_e_dist_sq = 0.0
#         self.log_count     = 0

#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)

#         self.timer = self.create_timer(self.Ts, self.control_loop)

#         self.get_logger().info('Node HA Nav2 Follower da san sang.')

#     def path_callback(self, msg: Path):
#         if len(msg.poses) < 2:
#             return
#         self.traj = Nav2Trajectory(msg)
#         if self.pose_received:
#             self.s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
#         else:
#             self.s = 0.0
#         self.v_s = 0.02
#         self.path_received = True
#         self.ctrl.reset()
        
#         # Reset biến tính toán
#         self.sum_e_dist_sq = 0.0
#         self.log_count = 0
        
#         self.get_logger().info(f'=== NHAN QUY DAO MOI: {self.traj.L_total:.2f} m ===')

#     def _v_safe(self, s):
#         v_min = self.v_path_max
#         n = max(self.num_curve_samp, 1)
#         for i in range(n + 1):
#             kap  = self.traj.curvature(s + i / n * self.look_ahead_dist)
#             v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
#             if v_ok < v_min:
#                 v_min = v_ok
#         return max(v_min, 0.03)

#     def control_loop(self):
#         try:
#             now = rclpy.time.Time()
#             tf_trans = self.tf_buffer.lookup_transform('map', 'base_link', now)
#             rx = tf_trans.transform.translation.x
#             ry = tf_trans.transform.translation.y
#             q  = tf_trans.transform.rotation
#             rphi = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
#             self.curr_q = [rx, ry, rphi]
#             self.pose_received = True
#         except TransformException:
#             return

#         if not self.pose_received or not self.path_received or self.traj is None:
#             return

#         v_adapt    = self._v_safe(self.s)
#         dist_brake = self.v_s**2 / (2 * self.a_path_max + 1e-9)
#         dist_left  = self.traj.L_total - self.s

#         if dist_left <= dist_brake:
#             a_s = -self.a_path_max
#         elif self.v_s < v_adapt - 0.01:
#             a_s =  self.a_path_max
#         elif self.v_s > v_adapt + 0.01:
#             a_s = -self.a_path_max
#         else:
#             a_s =  0.0

#         self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.v_path_max))
#         self.s  += self.v_s * self.Ts

#         nearest_s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
#         if self.s > nearest_s + self.look_ahead_dist:
#             self.s = nearest_s + self.look_ahead_dist

#         if self.s >= self.traj.L_total - 0.05:
#             self.stop_robot()
#             self.get_logger().info('=== HOAN THANH QUY DAO NAV2 ===')
#             self.path_received = False
#             return

#         phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
#         x_ref, y_ref = self.traj.pos(self.s)

#         rx, ry, rphi = self.curr_q
#         dx_g = x_ref - rx
#         dy_g = y_ref - ry
#         cq = math.cos(rphi)
#         sq = math.sin(rphi)

#         e1 =  cq * dx_g + sq * dy_g
#         e2 = -sq * dx_g + cq * dy_g
#         e3 = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))

#         v_cmd, w_cmd, _ = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

#         cmd = Twist()
#         cmd.linear.x  = v_cmd
#         cmd.angular.z = w_cmd
#         self.cmd_pub.publish(cmd)

#         # --- TÍNH TOÁN RMSE & IN LOG TEXT THUẦN ---
#         e_dist = math.hypot(dx_g, dy_g)
#         self.sum_e_dist_sq += e_dist**2
#         self.log_count += 1
#         rmse = math.sqrt(self.sum_e_dist_sq / self.log_count)

#         # In ra màn hình mỗi 10 vòng lặp (2 lần/giây)
#         if self.log_count % 10 == 0:
#             msg = (
#                 f"v_cmd: {v_cmd:.3f} | w_cmd: {w_cmd:+.3f} || "
#                 f"e1: {e1:+.3f} | e2: {e2:+.3f} | e3: {math.degrees(e3):+05.1f} || "
#                 f"RMSE: {rmse:.3f}"
#             )
#             self.get_logger().info(msg)

#     def stop_robot(self):
#         self.cmd_pub.publish(Twist())


# def main():
#     rclpy.init()
#     node = HANav2ExpertNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Ctrl+C nhan duoc, dung robot.')
#     finally:
#         node.stop_robot()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
# """
# ha_nav2_expert_follower.py
# Bộ điều khiển bám quỹ đạo HA 12 Terms - Chuyên gia điều khiển
# Giữ nguyên thuật toán gốc (Kanayama, LPF, v_safe, virtual target, feedforward)
# Cấu trúc ROS2 được đơn giản hóa theo phong cách code thứ hai.
# """

# import math
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Path
# from tf2_ros.buffer import Buffer
# from tf2_ros.transform_listener import TransformListener
# from tf2_ros import TransformException


# # ===========================================================================
# #  HEDGE ALGEBRA (12 ngữ nghĩa - Giữ nguyên hoàn toàn)
# # ===========================================================================
# class HedgeAlgebra:
#     def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
#         self.theta = theta
#         self.alpha = alpha
#         self.beta  = beta
#         self.fm    = {'V': alpha, 'L': beta, 'Small': theta, 'Large': 1.0 - theta}

#         self.term_specs = [
#             ('Zero',     [],          None   ),
#             ('VVSmall',  ['V', 'V'],  'Small'),
#             ('VSmall',   ['V'],       'Small'),
#             ('Small',    [],          'Small'),
#             ('VLSmall',  ['V', 'L'],  'Small'),
#             ('LSmall',   ['L'],       'Small'),
#             ('Neutral',  [],          None   ),
#             ('LLarge',   ['L'],       'Large'),
#             ('Large',    [],          'Large'),
#             ('VLarge',   ['V'],       'Large'),
#             ('VVLarge',  ['V', 'V'],  'Large'),
#             ('Absolute', [],          None   ),
#         ]

#         self.sem = {n: self._compute_sem(n, h, b) for n, h, b in self.term_specs}
#         self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])

#         self.hcnt = {
#             name: len(hedges) if name not in {'Zero', 'Neutral', 'Absolute', 'Small', 'Large'} else 0
#             for name, hedges, _ in self.term_specs
#         }

#     def _sgn_hedge(self, h):
#         return 1 if h == 'V' else -1

#     def _sgn_base(self, h, base):
#         if base == 'Small':
#             return -1 if h == 'V' else 1
#         else:
#             return  1 if h == 'V' else -1

#     def _compute_sem(self, name, hedges, base):
#         if name == 'Zero':     return 0.0
#         if name == 'Neutral':  return self.theta
#         if name == 'Absolute': return 1.0
#         fm_base   = self.fm[base]
#         current_v = (self.theta - self.alpha * fm_base if base == 'Small' else self.theta + self.alpha * fm_base)
#         if not hedges: return current_v
#         current_fm = fm_base
#         for h in reversed(hedges):
#             fm_hx  = (self.beta if h == 'V' else self.alpha) * current_fm
#             sum_fm = fm_hx if h == 'V' else current_fm
#             current_v += self._sgn_base(h, base) * (sum_fm - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha)) * fm_hx)
#             current_fm = fm_hx
#         return current_v

#     def fuzzify(self, value):
#         v     = float(np.clip(value, 0.0, 1.0))
#         min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
#         cands = [t for t in self.ordered_terms if abs(self.sem[t] - v) <= min_d + 1e-9]
#         return min(cands, key=lambda t: self.hcnt.get(t, 3))

#     def infer(self, input_val, rule_base):
#         term = self.fuzzify(float(np.clip(input_val, 0.0, 1.0)))
#         return self.sem[rule_base.get(term, 'Neutral')]


# # ===========================================================================
# #  RULE BASES (Giữ nguyên)
# # ===========================================================================
# def _rule_longitudinal():
#     return {'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero', 'Small': 'VSmall', 'VLSmall': 'Small', 
#             'LSmall': 'VLSmall', 'Neutral': 'LSmall', 'LLarge': 'Large', 'Large': 'Large', 
#             'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}

# def _rule_lateral():
#     return {'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero', 'Small': 'VSmall', 'VLSmall': 'Small', 
#             'LSmall': 'LSmall', 'Neutral': 'Large', 'LLarge': 'VLarge', 'Large': 'VLarge', 
#             'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}

# def _rule_heading():
#     return {'Zero': 'Zero', 'VVSmall': 'VVSmall', 'VSmall': 'VSmall', 'Small': 'Small', 'VLSmall': 'VLSmall', 
#             'LSmall': 'LSmall', 'Neutral': 'Large', 'LLarge': 'LLarge', 'Large': 'VLarge', 
#             'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'}


# # ===========================================================================
# #  HA CONTROLLER (Giữ nguyên hoàn toàn)
# # ===========================================================================
# class HAController:
#     MAX_E1 = 1.0
#     MAX_E2 = 1.0
#     MAX_E3 = math.pi / 3

#     def __init__(self, ha, gain_e1=10.0, gain_e2=15.0, gain_e3=15.0,
#                  v_max=0.15, w_max=2.0, dv_max=1.0, dw_max=1.0, Ts=0.05, lpf_alpha=0.65):
#         self.ha         = ha
#         self.gain_e1    = gain_e1
#         self.gain_e2    = gain_e2
#         self.gain_e3    = gain_e3
#         self.v_max      = v_max
#         self.w_max      = w_max
#         self.dv_max     = dv_max
#         self.dw_max     = dw_max
#         self.Ts         = Ts
#         self.lpf_alpha  = lpf_alpha
#         self.rule_long  = _rule_longitudinal()
#         self.rule_lat   = _rule_lateral()
#         self.rule_head  = _rule_heading()
#         self.v_prev     = 0.0
#         self.w_prev     = 0.0
#         self.w_lpf_prev = 0.0

#     def _ha_fb(self, err, max_e, rule, gain):
#         if abs(err) <= 0.01 * max_e: return 0.0
#         norm = min(abs(err) / max_e, 1.0)
#         mag  = self.ha.infer(norm, rule) * gain
#         return math.copysign(mag, err)

#     def compute(self, v_ref, w_ref, e1, e2, e3):
#         fb1 = self._ha_fb(e1, self.MAX_E1, self.rule_long, self.gain_e1)
#         fb2 = self._ha_fb(e2, self.MAX_E2, self.rule_lat,  self.gain_e2)
#         fb3 = self._ha_fb(e3, self.MAX_E3, self.rule_head, self.gain_e3)

#         v_raw = v_ref * math.cos(e3) + fb1
#         w_raw = w_ref + fb2 + fb3

#         w_filt = self.lpf_alpha * self.w_lpf_prev + (1 - self.lpf_alpha) * w_raw
#         self.w_lpf_prev = w_filt

#         v_sat = float(np.clip(v_raw,  -self.v_max, self.v_max))
#         w_sat = float(np.clip(w_filt, -self.w_max, self.w_max))

#         dt  = self.Ts
#         dv  = float(np.clip(v_sat - self.v_prev, -self.dv_max * dt, self.dv_max * dt))
#         dw  = float(np.clip(w_sat - self.w_prev, -self.dw_max * dt, self.dw_max * dt))

#         v_cmd = self.v_prev + dv
#         w_cmd = self.w_prev + dw
#         self.v_prev = v_cmd
#         self.w_prev = w_cmd
#         return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2, 'fb3': fb3}

#     def reset(self):
#         self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


# # ===========================================================================
# #  NAV2 TRAJECTORY (xử lý path từ Nav2)
# # ===========================================================================
# class Nav2Trajectory:
#     def __init__(self, path_msg: Path):
#         self.xs = np.array([p.pose.position.x for p in path_msg.poses])
#         self.ys = np.array([p.pose.position.y for p in path_msg.poses])
        
#         if len(self.xs) < 2:
#             self.s_arr = np.array([0.0])
#             self.L_total = 0.0
#             return

#         dx = np.diff(self.xs)
#         dy = np.diff(self.ys)
#         ds = np.hypot(dx, dy)
#         self.s_arr = np.concatenate([[0.0], np.cumsum(ds)])
#         self.L_total = self.s_arr[-1]

#         # Tính hướng (phi)
#         self.phis = np.zeros_like(self.xs)
#         for i in range(len(self.xs) - 1):
#             self.phis[i] = math.atan2(self.ys[i+1] - self.ys[i], self.xs[i+1] - self.xs[i])
#         self.phis[-1] = self.phis[-2]
#         self.phis = np.unwrap(self.phis)

#         # Tính độ cong (curvature)
#         self.kappas = np.zeros_like(self.xs)
#         for i in range(1, len(self.xs) - 1):
#             ds_val = self.s_arr[i+1] - self.s_arr[i-1]
#             if ds_val > 1e-4:
#                 self.kappas[i] = (self.phis[i+1] - self.phis[i-1]) / ds_val

#     def pos(self, s):
#         s = np.clip(s, 0.0, self.L_total)
#         x = float(np.interp(s, self.s_arr, self.xs))
#         y = float(np.interp(s, self.s_arr, self.ys))
#         return x, y

#     def curvature(self, s):
#         s = np.clip(s, 0.0, self.L_total)
#         return abs(float(np.interp(s, self.s_arr, self.kappas)))

#     def ref_kinematics(self, s, v_s, a_s):
#         s = np.clip(s, 0.0, self.L_total)
#         phi_ref = float(np.interp(s, self.s_arr, self.phis))
#         kappa = float(np.interp(s, self.s_arr, self.kappas))
#         v_ref = v_s
#         w_ref = kappa * v_s
#         return phi_ref, v_ref, w_ref

#     def find_nearest_s(self, rx, ry):
#         if self.L_total == 0.0: return 0.0
#         dists = np.hypot(self.xs - rx, self.ys - ry)
#         return float(self.s_arr[np.argmin(dists)])


# # ===========================================================================
# #  MAIN ROS2 NODE - CẤU TRÚC ĐƠN GIẢN (giống code thứ hai)
# # ===========================================================================
# class HANav2ExpertNode(Node):

#     def __init__(self):
#         super().__init__('ha_nav2_expert')

#         # Tham số (đơn giản, không use_sim_time, không parameter_overrides)
#         self.declare_parameter('Ts', 0.05)
#         self.declare_parameter('v_path_max', 0.4)
#         self.declare_parameter('a_path_max', 0.2)

#         self.Ts         = self.get_parameter('Ts').value
#         self.v_path_max = self.get_parameter('v_path_max').value
#         self.a_path_max = self.get_parameter('a_path_max').value

#         # Tham số điều khiển (hardcode giống bản gốc, có thể điều chỉnh nếu cần)
#         self.look_ahead_dist = 0.30
#         self.curve_beta      = 0.06
#         self.num_curve_samp  = 8

#         self.get_logger().info(f'HA 12 Terms | Ts={self.Ts}s | v_max={self.v_path_max}m/s')

#         # Đối tượng xử lý quỹ đạo và bộ điều khiển
#         self.traj = None
#         self.ha   = HedgeAlgebra()
#         self.ctrl = HAController(
#             ha=self.ha,
#             gain_e1=0.8, gain_e2=1.20, gain_e3=1.20,
#             v_max=0.15,    w_max=2.0,
#             dv_max=2.00,  dw_max=10.00,
#             Ts=self.Ts,   lpf_alpha=0.65)

#         # Trạng thái
#         self.s             = 0.0   # virtual target
#         self.v_s           = 0.02
#         self.curr_q        = [0.0, 0.0, 0.0]
#         self.pose_received = False
#         self.path_received = False

#         # ROS2: TF thay vì odom
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # Publisher / Subscriber (QoS đơn giản)
#         self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)

#         # Timer điều khiển
#         self.timer = self.create_timer(self.Ts, self.control_loop)

#         self.get_logger().info('Node HA Nav2 Follower đã sẵn sàng (cấu trúc đơn giản).')

#     def path_callback(self, msg: Path):
#         if len(msg.poses) < 2:
#             return
#         self.traj = Nav2Trajectory(msg)
#         if self.pose_received:
#             self.s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
#         else:
#             self.s = 0.0
#         self.v_s = 0.02
#         self.path_received = True
#         self.ctrl.reset()
#         self.get_logger().info(f'Nhận quỹ đạo mới: {self.traj.L_total:.2f} m')

#     def _v_safe(self, s):
#         v_min = self.v_path_max
#         n = max(self.num_curve_samp, 1)
#         for i in range(n + 1):
#             kap  = self.traj.curvature(s + i / n * self.look_ahead_dist)
#             v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
#             if v_ok < v_min:
#                 v_min = v_ok
#         return max(v_min, 0.03)

#     def control_loop(self):
#         # 1. Lấy pose từ TF
#         try:
#             now = rclpy.time.Time()
#             tf_trans = self.tf_buffer.lookup_transform('map', 'base_link', now)
#             rx = tf_trans.transform.translation.x
#             ry = tf_trans.transform.translation.y
#             q  = tf_trans.transform.rotation
#             rphi = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
#             self.curr_q = [rx, ry, rphi]
#             self.pose_received = True
#         except TransformException:
#             return

#         if not self.pose_received or not self.path_received or self.traj is None:
#             return

#         # 2. Cập nhật virtual target s và v_s (thuật toán gốc)
#         v_adapt    = self._v_safe(self.s)
#         dist_brake = self.v_s**2 / (2 * self.a_path_max + 1e-9)
#         dist_left  = self.traj.L_total - self.s

#         if dist_left <= dist_brake:
#             a_s = -self.a_path_max
#         elif self.v_s < v_adapt - 0.01:
#             a_s =  self.a_path_max
#         elif self.v_s > v_adapt + 0.01:
#             a_s = -self.a_path_max
#         else:
#             a_s =  0.0

#         self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.v_path_max))
#         self.s  += self.v_s * self.Ts

#         # Kỹ thuật khóa virtual target (chống tụt lại)
#         nearest_s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
#         if self.s > nearest_s + self.look_ahead_dist:
#             self.s = nearest_s + self.look_ahead_dist

#         # Kiểm tra hoàn thành
#         if self.s >= self.traj.L_total - 0.05:
#             self.stop_robot()
#             self.get_logger().info('=== Hoàn thành quỹ đạo Nav2 ===')
#             self.path_received = False
#             return

#         # 3. Tính toán sai số Kanayama
#         phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
#         x_ref, y_ref = self.traj.pos(self.s)

#         rx, ry, rphi = self.curr_q
#         dx_g = x_ref - rx
#         dy_g = y_ref - ry
#         cq = math.cos(rphi)
#         sq = math.sin(rphi)

#         e1 =  cq * dx_g + sq * dy_g
#         e2 = -sq * dx_g + cq * dy_g
#         e3 = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))

#         # 4. Tính v_cmd, w_cmd qua bộ HA Controller
#         v_cmd, w_cmd, _ = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

#         # 5. Publish lệnh
#         cmd = Twist()
#         cmd.linear.x  = v_cmd
#         cmd.angular.z = w_cmd
#         self.cmd_pub.publish(cmd)

#         # Log đơn giản (không CSV, không RMS)
#         e_dist = math.hypot(dx_g, dy_g)
#         self.get_logger().debug(f'es={e_dist:.3f}m | e2={e2:.3f}m | e3={math.degrees(e3):.1f}deg | v={v_cmd:.3f} w={w_cmd:.3f}')

#     def stop_robot(self):
#         # Dừng khẩn cấp
#         self.cmd_pub.publish(Twist())


# def main():
#     rclpy.init()
#     node = HANav2ExpertNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Ctrl+C nhận được, dừng robot.')
#     finally:
#         node.stop_robot()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()