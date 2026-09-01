#!/usr/bin/env python3
"""
ha_nav2_expert_follower_v4_final.py
Bộ điều khiển bám quỹ đạo HA 12 Terms - Bản chuẩn chuyên gia (Tích hợp Log giám sát)

TỔNG HỢP CÁC TÍNH NĂNG & FIX LỖI:
  - [BUG 1-6]: Hiệu chỉnh vùng sai số, clamp tốc độ virtual target, giảm dw_max, thêm LPF.
  - [BUG 7]: Sửa lỗi dừng non (Dùng tọa độ thực tế để check đích).
  - [BUG 8]: Khử nhiễu góc quay (Set w_ref = 0.0 để tránh nhiễu do lưới path).
  - [BUG 9]: Cơ chế In-place Rotation (Xoay tại chỗ nếu góc lệch > 45 độ).
  - [FEATURE]: Tích hợp Log hiển thị sai số CTE, e1, e2, e3 và RMS (10 chu kỳ/lần).
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
#  HEDGE ALGEBRA (12 ngữ nghĩa)
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

    def _sgn_hedge(self, h):
        return 1 if h == 'V' else -1

    def _sgn_base(self, h, base):
        if base == 'Small':
            return -1 if h == 'V' else 1
        else:
            return  1 if h == 'V' else -1

    def _compute_sem(self, name, hedges, base):
        if name == 'Zero':     return 0.0
        if name == 'Neutral':  return self.theta
        if name == 'Absolute': return 1.0
        fm_base   = self.fm[base]
        current_v = (self.theta - self.alpha * fm_base if base == 'Small'
                     else self.theta + self.alpha * fm_base)
        if not hedges:
            return current_v
        current_fm = fm_base
        for h in reversed(hedges):
            fm_hx      = (self.beta if h == 'V' else self.alpha) * current_fm
            sum_fm     = fm_hx if h == 'V' else current_fm
            current_v += self._sgn_base(h, base) * (
                sum_fm - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha)) * fm_hx)
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


def _rule_longitudinal():
    return {
        'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero',
        'Small': 'VSmall', 'VLSmall': 'Small', 'LSmall': 'VLSmall',
        'Neutral': 'LSmall', 'LLarge': 'Large', 'Large': 'Large',
        'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'
    }

def _rule_lateral():
    return {
        'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero',
        'Small': 'VSmall', 'VLSmall': 'Small', 'LSmall': 'LSmall',
        'Neutral': 'Large', 'LLarge': 'VLarge', 'Large': 'VLarge',
        'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'
    }

def _rule_heading():
    return {
        'Zero': 'Zero', 'VVSmall': 'VVSmall', 'VSmall': 'VSmall',
        'Small': 'Small', 'VLSmall': 'VLSmall', 'LSmall': 'LSmall',
        'Neutral': 'Large', 'LLarge': 'LLarge', 'Large': 'VLarge',
        'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'
    }


# ===========================================================================
#  HA CONTROLLER
# ===========================================================================
class HAController:
    MAX_E1 = 0.1  
    MAX_E2 = 0.1   
    MAX_E3 = math.pi / 4  

    DEADBAND_E1 = 0.005   
    DEADBAND_E2 = 0.005   
    DEADBAND_E3 = 0.008   
    DEADBAND_W  = 0.02    

    def __init__(self, ha, gain_e1=0.25, gain_e2=0.70, gain_e3=1.2,
                 v_max=0.15, w_max=1.5, dv_max=0.8, dw_max=4.0, 
                 Ts=0.05, lpf_alpha=0.80):
        self.ha          = ha
        self.gain_e1     = gain_e1
        self.gain_e2     = gain_e2
        self.gain_e3     = gain_e3
        self.v_max       = v_max
        self.w_max       = w_max
        self.dv_max      = dv_max
        self.dw_max      = dw_max
        self.Ts          = Ts
        self.lpf_alpha   = lpf_alpha
        self.rule_long   = _rule_longitudinal()
        self.rule_lat    = _rule_lateral()
        self.rule_head   = _rule_heading()
        self.v_prev      = 0.0
        self.w_prev      = 0.0
        self.w_lpf_prev  = 0.0

    def _ha_fb(self, err, max_e, deadband, rule, gain):
        if abs(err) <= deadband:
            return 0.0
        norm = min(abs(err) / max_e, 1.0)
        mag  = self.ha.infer(norm, rule) * gain
        return math.copysign(mag, err)

    def compute(self, v_ref, w_ref, e1, e2, e3):
        fb1 = self._ha_fb(e1, self.MAX_E1, self.DEADBAND_E1, self.rule_long, self.gain_e1)
        fb2 = self._ha_fb(e2, self.MAX_E2, self.DEADBAND_E2, self.rule_lat,  self.gain_e2)
        fb3 = self._ha_fb(e3, self.MAX_E3, self.DEADBAND_E3, self.rule_head, self.gain_e3)

        heading_saturation = min(abs(e3) / (math.pi / 6), 1.0) 
        fb2_scaled = fb2 * (1.0 - 0.6 * heading_saturation)

        v_raw  = v_ref * math.cos(e3) + fb1
        w_raw  = w_ref + fb2_scaled + fb3

        # In-place rotation (Cơ chế xoay tại chỗ)
        if abs(e3) > math.radians(45.0):
            v_raw = 0.0  
            w_raw = fb3  

        # LPF
        w_filt = self.lpf_alpha * self.w_lpf_prev + (1.0 - self.lpf_alpha) * w_raw
        self.w_lpf_prev = w_filt

        # Saturation
        v_sat = float(np.clip(v_raw,  -self.v_max,  self.v_max))
        w_sat = float(np.clip(w_filt, -self.w_max,  self.w_max))

        # Rate limiting 
        dt    = self.Ts
        dv    = float(np.clip(v_sat - self.v_prev, -self.dv_max * dt, self.dv_max * dt))
        dw    = float(np.clip(w_sat - self.w_prev, -self.dw_max * dt, self.dw_max * dt))

        v_cmd = self.v_prev + dv
        w_cmd = self.w_prev + dw

        if abs(w_cmd) < self.DEADBAND_W:
            w_cmd = 0.0

        self.v_prev = v_cmd
        self.w_prev = w_cmd
        return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2_scaled, 'fb3': fb3}

    def reset(self):
        self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


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
            self.phis[i] = math.atan2(self.ys[i+1] - self.ys[i],
                                      self.xs[i+1] - self.xs[i])
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

    def curvature(self, s):
        s = np.clip(s, 0.0, self.L_total)
        return abs(float(np.interp(s, self.s_arr, self.kappas)))

    def ref_kinematics(self, s, v_s):
        s       = np.clip(s, 0.0, self.L_total)
        phi_ref = float(np.interp(s, self.s_arr, self.phis))
        v_ref   = v_s
        
        # Khử nhiễu giật góc (Curvature Noise)
        w_ref   = 0.0  
        return phi_ref, v_ref, w_ref

    def find_nearest_s(self, rx, ry):
        if self.L_total == 0.0:
            return 0.0
        dists = np.hypot(self.xs - rx, self.ys - ry)
        return float(self.s_arr[np.argmin(dists)])

    def get_cross_track_error(self, rx, ry):
        if self.L_total == 0.0:
            return 0.0, 0.0
        nearest_idx = np.argmin(np.hypot(self.xs - rx, self.ys - ry))
        nearest_s   = self.s_arr[nearest_idx]
        x_n, y_n   = self.pos(nearest_s)
        cte         = math.hypot(rx - x_n, ry - y_n)
        return cte, nearest_s


# ===========================================================================
#  MAIN ROS2 NODE
# ===========================================================================
class HANav2ExpertNode(Node):

    LA_K    = 2.5    
    LA_MIN  = 0.20   
    LA_MAX  = 0.50   

    def __init__(self):
        super().__init__('ha_nav2_expert')

        self.declare_parameter('Ts',         0.05)
        self.declare_parameter('v_path_max', 0.25)   
        self.declare_parameter('a_path_max', 0.20)

        self.Ts         = self.get_parameter('Ts').value
        self.v_path_max = self.get_parameter('v_path_max').value
        self.a_path_max = self.get_parameter('a_path_max').value

        self.curve_beta     = 0.10
        self.num_curve_samp = 10
        self.stop_zone      = 0.08   

        self.get_logger().info(
            f'HA v4 Final | Ts={self.Ts}s | v_path_max={self.v_path_max}m/s | a_path_max={self.a_path_max}m/s2')

        self.traj = None
        self.ha   = HedgeAlgebra()

        self.ctrl = HAController(
            ha=self.ha, gain_e1=0.25, gain_e2=0.70, gain_e3=1.20,
            v_max=self.v_path_max, w_max=1.5, dv_max=0.8, dw_max=4.0,
            Ts=self.Ts, lpf_alpha=0.80 
        )

        self.s             = 0.0
        self.v_s           = 0.01
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

        self.get_logger().info('Node HA Nav2 Follower v4 Final da san sang.')

    def _look_ahead(self):
        return float(np.clip(self.LA_K * self.v_s, self.LA_MIN, self.LA_MAX))

    def path_callback(self, msg: Path):
        if len(msg.poses) < 2:
            return
        self.traj = Nav2Trajectory(msg)
        self.path_received = True

        if self.pose_received:
            self.s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
            self.s_initialized = True
        else:
            self.s = 0.0
            self.s_initialized = False

        self.v_s = 0.01
        self.ctrl.reset()
        self.sum_cross_sq = 0.0
        self.max_cross    = 0.0
        self.log_count    = 0
        self.get_logger().info(f'=== QUY DAO MOI: {self.traj.L_total:.2f}m ===')

    def _v_safe(self, s):
        if self.traj is None:
            return self.v_path_max
        la   = self._look_ahead()
        n    = max(self.num_curve_samp, 1)
        vmin = self.v_path_max
        for i in range(n + 1):
            kap  = self.traj.curvature(s + i / n * la)
            v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
            if v_ok < vmin:
                vmin = v_ok
        return max(vmin, 0.02)

    def control_loop(self):
        try:
            tf_trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            rx   = tf_trans.transform.translation.x
            ry   = tf_trans.transform.translation.y
            q    = tf_trans.transform.rotation
            rphi = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y**2 + q.z**2))
            self.curr_q      = [rx, ry, rphi]
            self.pose_received = True
        except TransformException:
            return

        if not self.path_received or self.traj is None:
            return

        rx, ry, rphi = self.curr_q
        if not self.s_initialized:
            self.s = self.traj.find_nearest_s(rx, ry)
            self.s_initialized = True

        v_adapt = min(self._v_safe(self.s), self.ctrl.v_max)

        dist_left  = self.traj.L_total - self.s
        dist_brake = self.v_s**2 / (2.0 * self.a_path_max + 1e-9)

        if dist_left <= dist_brake:
            a_s = -self.a_path_max
        elif self.v_s < v_adapt - 0.005:
            a_s =  self.a_path_max
        elif self.v_s > v_adapt + 0.005:
            a_s = -self.a_path_max
        else:
            a_s = 0.0

        self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.ctrl.v_max))
        self.s  += self.v_s * self.Ts

        cte, nearest_s = self.traj.get_cross_track_error(rx, ry)
        la = self._look_ahead()
        if self.s > nearest_s + la:
            self.s = nearest_s + la
            
        self.s = min(self.s, self.traj.L_total)

        # Kiểm tra điều kiện dừng
        x_end, y_end = self.traj.xs[-1], self.traj.ys[-1]
        dist_to_goal = math.hypot(rx - x_end, ry - y_end)

        if dist_to_goal <= self.stop_zone:
            self._stop_robot()
            rms = math.sqrt(self.sum_cross_sq / max(self.log_count, 1))
            self.get_logger().info(
                f'=== HOAN THANH TAI DICH: RMS_cross={rms:.4f}m | max_cross={self.max_cross:.4f}m ===')
            self.path_received = False
            return

        phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s)
        x_ref, y_ref          = self.traj.pos(self.s)

        dx_g = x_ref - rx
        dy_g = y_ref - ry
        cq   = math.cos(rphi)
        sq   = math.sin(rphi)

        e1   =  cq * dx_g + sq * dy_g   
        e2   = -sq * dx_g + cq * dy_g   
        e3   = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))  

        v_cmd, w_cmd, dbg = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

        cmd             = Twist()
        cmd.linear.x    = v_cmd
        cmd.angular.z   = w_cmd
        self.cmd_pub.publish(cmd)

        # Cập nhật thông số giám sát
        self.sum_cross_sq += cte**2
        self.log_count    += 1
        if cte > self.max_cross:
            self.max_cross = cte

        # In log ra terminal (10 chu kỳ / lần)
        rms_cross = math.sqrt(self.sum_cross_sq / self.log_count)
        
        if self.log_count % 10 == 0:
            self.get_logger().info(
                f"v={v_cmd:.3f} w={w_cmd:+.3f} | "
                f"CTE={cte:.3f} along={nearest_s - self.s:+.3f} | "
                f"e1={e1:+.3f} e2={e2:+.3f} e3={math.degrees(e3):+05.1f}deg | "
                f"RMS={rms_cross:.4f} max={self.max_cross:.4f}"
            )

    def _stop_robot(self):
        stop = Twist()
        self.cmd_pub.publish(stop)
        for _ in range(3):
            self.cmd_pub.publish(stop)

def main(args=None):
    rclpy.init(args=args)
    node = HANav2ExpertNode() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
    # """
# ha_nav2_expert_follower_v3.py
# Bộ điều khiển bám quỹ đạo HA 12 Terms - Bản sửa lỗi toàn diện

# CÁC LỖI ĐÃ SỬA:
#   [BUG-1 CRITICAL] MAX_E1=MAX_E2=10.0 → sai số 0.3m bị normalize=0.03 → vào vùng Zero/VVSmall
#                    → HA gần như không ra tín hiệu sửa lỗi gì → xe lệch mà không tự chỉnh.
#                    FIX: MAX_E1=0.50m, MAX_E2=0.40m (phù hợp dải sai số thực tế của robot nhỏ)

#   [BUG-2 CRITICAL] v_path_max=0.4 >> v_max_ctrl=0.15 → virtual target chạy trước robot 2.7x
#                    → e1 tích lũy lớn, robot "đuổi" target mãi, dẫn đến lắc/giật mạnh.
#                    FIX: virtual target speed được clip theo v_max_ctrl, thêm clamping mềm.

#   [BUG-3 HIGH]     dw_max=10.0 rad/s² → tốc độ góc thay đổi ±0.5 rad/s mỗi tick 20ms
#                    → robot quay giật mạnh trên mỗi chu kỳ điều khiển.
#                    FIX: dw_max=4.0, đồng thời tăng lpf_alpha=0.80 để lọc nhiễu w.

#   [BUG-4 MEDIUM]   e2 và e3 đều cộng vào w_raw mà không có trọng số cân bằng
#                    → ở góc quẹo: fb2+fb3 có thể bão hoà w_max liên tục → dao động.
#                    FIX: Tách trọng số, e3 ưu tiên hơn e2 khi |e3| lớn.

#   [BUG-5 MEDIUM]   look_ahead_dist cố định 0.30m không phụ thuộc tốc độ
#                    → ở tốc độ thấp: target quá xa; ở tốc độ cao: target quá gần.
#                    FIX: look_ahead_dist thích nghi = clip(k_la * v_s, la_min, la_max)

#   [BUG-6 LOW]      Không có deadband cho w_cmd nhỏ → micro-oscillation liên tục.
#                    FIX: deadband 0.02 rad/s cho w_cmd cuối.
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
# #  HEDGE ALGEBRA (12 ngữ nghĩa) - KHÔNG THAY ĐỔI
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
#         current_v = (self.theta - self.alpha * fm_base if base == 'Small'
#                      else self.theta + self.alpha * fm_base)
#         if not hedges:
#             return current_v
#         current_fm = fm_base
#         for h in reversed(hedges):
#             fm_hx      = (self.beta if h == 'V' else self.alpha) * current_fm
#             sum_fm     = fm_hx if h == 'V' else current_fm
#             current_v += self._sgn_base(h, base) * (
#                 sum_fm - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha)) * fm_hx)
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
# #  RULE BASES - KHÔNG THAY ĐỔI
# # ===========================================================================
# def _rule_longitudinal():
#     return {
#         'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero',
#         'Small': 'VSmall', 'VLSmall': 'Small', 'LSmall': 'VLSmall',
#         'Neutral': 'LSmall', 'LLarge': 'Large', 'Large': 'Large',
#         'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'
#     }

# def _rule_lateral():
#     return {
#         'Zero': 'Zero', 'VVSmall': 'Zero', 'VSmall': 'Zero',
#         'Small': 'VSmall', 'VLSmall': 'Small', 'LSmall': 'LSmall',
#         'Neutral': 'Large', 'LLarge': 'VLarge', 'Large': 'VLarge',
#         'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'
#     }

# def _rule_heading():
#     return {
#         'Zero': 'Zero', 'VVSmall': 'VVSmall', 'VSmall': 'VSmall',
#         'Small': 'Small', 'VLSmall': 'VLSmall', 'LSmall': 'LSmall',
#         'Neutral': 'Large', 'LLarge': 'LLarge', 'Large': 'VLarge',
#         'VLarge': 'VVLarge', 'VVLarge': 'Absolute', 'Absolute': 'Absolute'
#     }


# # ===========================================================================
# #  HA CONTROLLER - SỬA BUG-1, BUG-3, BUG-4, BUG-6
# # ===========================================================================
# class HAController:
#     # [FIX BUG-1] Giảm MAX_E từ 10.0 → phù hợp dải lỗi thực tế robot nhỏ
#     # Tại v_max=0.15 m/s, sai số bình thường 0.02~0.20m:
#     #   MAX_E1=0.5m: norm=0.04~0.40 → vào vùng VVSmall~Small → HA phản ứng hợp lý
#     #   MAX_E2=0.4m: cross-track thường <0.15m → norm=0.04~0.38 → OK
#     MAX_E1 = 0.1  # [FIX] was 10.0 → HA gần như im lặng với lỗi nhỏ
#     MAX_E2 = 0.1   # [FIX] was 10.0
#     MAX_E3 = math.pi / 4  # 60 deg - giữ nguyên

#     # Deadband để tắt micro-correction [FIX BUG-6]
#     DEADBAND_E1 = 0.005   # 5mm
#     DEADBAND_E2 = 0.005   # 5mm
#     DEADBAND_E3 = 0.008   # ~0.5 deg
#     DEADBAND_W  = 0.02    # rad/s cuối

#     def __init__(self, ha,
#                  gain_e1=0.25, gain_e2=0.70, gain_e3=1.2,
#                  v_max=0.15, w_max=1.5,
#                  dv_max=0.8, dw_max=4.0,        # [FIX BUG-3] dw_max: 10→4
#                  Ts=0.05, lpf_alpha=0.80):       # [FIX BUG-3] lpf: 0.65→0.80
#         self.ha          = ha
#         self.gain_e1     = gain_e1
#         self.gain_e2     = gain_e2
#         self.gain_e3     = gain_e3
#         self.v_max       = v_max
#         self.w_max       = w_max
#         self.dv_max      = dv_max
#         self.dw_max      = dw_max
#         self.Ts          = Ts
#         self.lpf_alpha   = lpf_alpha
#         self.rule_long   = _rule_longitudinal()
#         self.rule_lat    = _rule_lateral()
#         self.rule_head   = _rule_heading()
#         self.v_prev      = 0.0
#         self.w_prev      = 0.0
#         self.w_lpf_prev  = 0.0

#     def _ha_fb(self, err, max_e, deadband, rule, gain):
#         """Feedback HA có deadband để triệt micro-oscillation."""
#         if abs(err) <= deadband:
#             return 0.0
#         norm = min(abs(err) / max_e, 1.0)
#         mag  = self.ha.infer(norm, rule) * gain
#         return math.copysign(mag, err)

#     def compute(self, v_ref, w_ref, e1, e2, e3):
#         """
#         [FIX BUG-4] Tách trọng số e2 và e3:
#           - Khi |e3| > 20 deg: giảm contribution e2 để tránh double-correction.
#           - Khi thẳng (|e3| nhỏ): e2 đóng vai trò chính chỉnh hướng về quỹ đạo.
#         """
#         fb1 = self._ha_fb(e1, self.MAX_E1, self.DEADBAND_E1, self.rule_long, self.gain_e1)
#         fb2 = self._ha_fb(e2, self.MAX_E2, self.DEADBAND_E2, self.rule_lat,  self.gain_e2)
#         fb3 = self._ha_fb(e3, self.MAX_E3, self.DEADBAND_E3, self.rule_head, self.gain_e3)

#         # [FIX BUG-4] Giảm fb2 khi đang cần quay mạnh (fb3 đã xử lý heading)
#         heading_saturation = min(abs(e3) / (math.pi / 6), 1.0)  # 0→1 khi |e3|→30°
#         fb2_scaled = fb2 * (1.0 - 0.6 * heading_saturation)

#         v_raw  = v_ref * math.cos(e3) + fb1
#         w_raw  = w_ref + fb2_scaled + fb3

#         # LPF cho w (alpha lớn hơn = mượt hơn) [FIX BUG-3]
#         w_filt = self.lpf_alpha * self.w_lpf_prev + (1.0 - self.lpf_alpha) * w_raw
#         self.w_lpf_prev = w_filt

#         # Saturation
#         v_sat = float(np.clip(v_raw,  -self.v_max,  self.v_max))
#         w_sat = float(np.clip(w_filt, -self.w_max,  self.w_max))

#         # Rate limiting [FIX BUG-3]
#         dt    = self.Ts
#         dv    = float(np.clip(v_sat - self.v_prev, -self.dv_max * dt, self.dv_max * dt))
#         dw    = float(np.clip(w_sat - self.w_prev, -self.dw_max * dt, self.dw_max * dt))

#         v_cmd = self.v_prev + dv
#         w_cmd = self.w_prev + dw

#         # [FIX BUG-6] Deadband cuối cho w để tắt micro-oscillation
#         if abs(w_cmd) < self.DEADBAND_W:
#             w_cmd = 0.0

#         self.v_prev = v_cmd
#         self.w_prev = w_cmd
#         return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2_scaled, 'fb3': fb3}

#     def reset(self):
#         self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


# # ===========================================================================
# #  NAV2 TRAJECTORY - THÊM SMOOTH CURVATURE
# # ===========================================================================
# class Nav2Trajectory:
#     def __init__(self, path_msg: Path):
#         self.xs = np.array([p.pose.position.x for p in path_msg.poses])
#         self.ys = np.array([p.pose.position.y for p in path_msg.poses])

#         if len(self.xs) < 2:
#             self.s_arr   = np.array([0.0])
#             self.L_total = 0.0
#             return

#         dx = np.diff(self.xs)
#         dy = np.diff(self.ys)
#         ds = np.hypot(dx, dy)
#         self.s_arr   = np.concatenate([[0.0], np.cumsum(ds)])
#         self.L_total = self.s_arr[-1]

#         # Hướng phi
#         self.phis = np.zeros_like(self.xs)
#         for i in range(len(self.xs) - 1):
#             self.phis[i] = math.atan2(self.ys[i+1] - self.ys[i],
#                                       self.xs[i+1] - self.xs[i])
#         self.phis[-1] = self.phis[-2]
#         self.phis = np.unwrap(self.phis)

#         # Độ cong kappa (tính bằng trung tâm sai phân)
#         self.kappas = np.zeros_like(self.xs)
#         for i in range(1, len(self.xs) - 1):
#             ds_val = self.s_arr[i+1] - self.s_arr[i-1]
#             if ds_val > 1e-4:
#                 self.kappas[i] = (self.phis[i+1] - self.phis[i-1]) / ds_val
#         # Smooth kappa bằng moving average để tránh curvature giật (do path discretization)
#         k = min(7, len(self.kappas))
#         self.kappas = np.convolve(self.kappas, np.ones(k) / k, mode='same')

#     def pos(self, s):
#         s = np.clip(s, 0.0, self.L_total)
#         return (float(np.interp(s, self.s_arr, self.xs)),
#                 float(np.interp(s, self.s_arr, self.ys)))

#     def curvature(self, s):
#         s = np.clip(s, 0.0, self.L_total)
#         return abs(float(np.interp(s, self.s_arr, self.kappas)))

#     def ref_kinematics(self, s, v_s):
#         s       = np.clip(s, 0.0, self.L_total)
#         phi_ref = float(np.interp(s, self.s_arr, self.phis))
#         kappa   = float(np.interp(s, self.s_arr, self.kappas))
#         v_ref   = v_s
#         w_ref   = kappa * v_s
#         return phi_ref, v_ref, w_ref

#     def find_nearest_s(self, rx, ry):
#         if self.L_total == 0.0:
#             return 0.0
#         dists = np.hypot(self.xs - rx, self.ys - ry)
#         return float(self.s_arr[np.argmin(dists)])

#     def get_cross_track_error(self, rx, ry):
#         """Cross-track error (khoảng cách vuông góc đến path)."""
#         if self.L_total == 0.0:
#             return 0.0, 0.0
#         nearest_idx = np.argmin(np.hypot(self.xs - rx, self.ys - ry))
#         nearest_s   = self.s_arr[nearest_idx]
#         x_n, y_n   = self.pos(nearest_s)
#         cte         = math.hypot(rx - x_n, ry - y_n)
#         return cte, nearest_s


# # ===========================================================================
# #  MAIN ROS2 NODE - SỬA BUG-2, BUG-5
# # ===========================================================================
# class HANav2ExpertNode(Node):

#     # Hằng số look-ahead [FIX BUG-5]
#     LA_K    = 2.5    # look_ahead = LA_K * v_s  (giây nhìn trước)
#     LA_MIN  = 0.20   # m - tối thiểu
#     LA_MAX  = 0.50   # m - tối đa

#     def __init__(self):
#         super().__init__('ha_nav2_expert')

#         self.declare_parameter('Ts',         0.05)
#         self.declare_parameter('v_path_max', 0.25)   # [FIX BUG-2] default giảm xuống khớp v_max_ctrl
#         self.declare_parameter('a_path_max', 0.20)

#         self.Ts         = self.get_parameter('Ts').value
#         self.v_path_max = self.get_parameter('v_path_max').value
#         self.a_path_max = self.get_parameter('a_path_max').value

#         # Curve-adaptive speed
#         self.curve_beta     = 0.10
#         self.num_curve_samp = 10

#         # Stop zone (dừng khi còn cách cuối path dưới ngưỡng này)
#         self.stop_zone = 0.08   # m

#         self.get_logger().info(
#             f'HA v3 | Ts={self.Ts}s | v_path_max={self.v_path_max}m/s | a_path_max={self.a_path_max}m/s2')

#         # Đối tượng chính
#         self.traj = None
#         self.ha   = HedgeAlgebra()

#         # [FIX] Tham số HAController đã được hiệu chỉnh lại
#         self.ctrl = HAController(
#             ha=self.ha,
#             gain_e1=0.25,   # longitudinal: nhẹ, chủ yếu do feedforward
#             gain_e2=0.70,   # lateral: đủ để kéo về quỹ đạo
#             gain_e3=1.20,   # heading: ưu tiên cao để xe quay sớm
#             v_max=self.v_path_max,
#             w_max=1.5,
#             dv_max=0.8,     # m/s² - mượt
#             dw_max=4.0,     # rad/s² [FIX BUG-3]
#             Ts=self.Ts,
#             lpf_alpha=0.80  # [FIX BUG-3]
#         )

#         # Trạng thái bộ điều khiển
#         self.s             = 0.0
#         self.v_s           = 0.01
#         self.curr_q        = [0.0, 0.0, 0.0]
#         self.pose_received = False
#         self.path_received = False
#         self.s_initialized = False

#         # Giám sát chất lượng bám
#         self.sum_cross_sq = 0.0
#         self.log_count    = 0
#         self.max_cross    = 0.0

#         # TF
#         self.tf_buffer   = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # ROS2 I/O
#         self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
#         self.timer    = self.create_timer(self.Ts, self.control_loop)

#         self.get_logger().info('Node HA Nav2 Follower v3 (bug-fixed) da san sang.')

#     # ------------------------------------------------------------------
#     def _look_ahead(self):
#         """[FIX BUG-5] Look-ahead thích nghi theo tốc độ hiện tại."""
#         return float(np.clip(self.LA_K * self.v_s, self.LA_MIN, self.LA_MAX))

#     def path_callback(self, msg: Path):
#         if len(msg.poses) < 2:
#             return
#         self.traj = Nav2Trajectory(msg)
#         self.path_received = True

#         if self.pose_received:
#             self.s = self.traj.find_nearest_s(self.curr_q[0], self.curr_q[1])
#             self.s_initialized = True
#             self.get_logger().info(f'Khoi tao s={self.s:.3f}m (pose da co)')
#         else:
#             self.s = 0.0
#             self.s_initialized = False
#             self.get_logger().info('Path truoc pose → s se khoi tao khi co pose.')

#         self.v_s = 0.01
#         self.ctrl.reset()
#         self.sum_cross_sq = 0.0
#         self.max_cross    = 0.0
#         self.log_count    = 0
#         self.get_logger().info(f'=== QUY DAO MOI: {self.traj.L_total:.2f}m ===')

#     # ------------------------------------------------------------------
#     def _v_safe(self, s):
#         """Tốc độ an toàn dựa theo độ cong nhìn trước."""
#         if self.traj is None:
#             return self.v_path_max
#         la   = self._look_ahead()
#         n    = max(self.num_curve_samp, 1)
#         vmin = self.v_path_max
#         for i in range(n + 1):
#             kap  = self.traj.curvature(s + i / n * la)
#             v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
#             if v_ok < vmin:
#                 vmin = v_ok
#         return max(vmin, 0.02)

#     # ------------------------------------------------------------------
#     def control_loop(self):
#         # ── 1. Lấy pose từ TF ─────────────────────────────────────────
#         try:
#             tf_trans = self.tf_buffer.lookup_transform(
#                 'map', 'base_link', rclpy.time.Time())
#             rx   = tf_trans.transform.translation.x
#             ry   = tf_trans.transform.translation.y
#             q    = tf_trans.transform.rotation
#             rphi = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
#                               1.0 - 2.0 * (q.y**2 + q.z**2))
#             self.curr_q      = [rx, ry, rphi]
#             self.pose_received = True
#         except TransformException:
#             return

#         if not self.path_received or self.traj is None:
#             return

#         # ── 2. Đồng bộ s lần đầu ──────────────────────────────────────
#         rx, ry, rphi = self.curr_q
#         if not self.s_initialized:
#             self.s = self.traj.find_nearest_s(rx, ry)
#             self.s_initialized = True
#             self.get_logger().info(f'Khoi tao s={self.s:.3f}m (pose vua nhan)')

#         # ── 3. Cập nhật v_s và s (virtual target) ─────────────────────
#         v_adapt = self._v_safe(self.s)

#         # [FIX BUG-2] Đảm bảo v_s không bao giờ lớn hơn v_max của controller
#         v_adapt = min(v_adapt, self.ctrl.v_max)

#         dist_left  = self.traj.L_total - self.s
#         dist_brake = self.v_s**2 / (2.0 * self.a_path_max + 1e-9)

#         if dist_left <= dist_brake:
#             a_s = -self.a_path_max
#         elif self.v_s < v_adapt - 0.005:
#             a_s =  self.a_path_max
#         elif self.v_s > v_adapt + 0.005:
#             a_s = -self.a_path_max
#         else:
#             a_s = 0.0

#         self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.ctrl.v_max))
#         self.s  += self.v_s * self.Ts

#         # ── 4. Clamp s không vượt quá look-ahead so với robot ─────────
#         cte, nearest_s = self.traj.get_cross_track_error(rx, ry)
#         la = self._look_ahead()
#         if self.s > nearest_s + la:
#             self.s = nearest_s + la

#         # ── 5. Kiểm tra kết thúc ──────────────────────────────────────
#         if self.s >= self.traj.L_total - self.stop_zone:
#             self._stop_robot()
#             rms = math.sqrt(self.sum_cross_sq / max(self.log_count, 1))
#             self.get_logger().info(
#                 f'=== HOAN THANH: RMS_cross={rms:.4f}m | max_cross={self.max_cross:.4f}m ===')
#             self.path_received = False
#             return

#         # ── 6. Tính kinematics tham chiếu tại virtual target ──────────
#         phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s)
#         x_ref, y_ref          = self.traj.pos(self.s)

#         # ── 7. Sai số Kanayama trong khung robot ──────────────────────
#         dx_g = x_ref - rx
#         dy_g = y_ref - ry
#         cq   = math.cos(rphi)
#         sq   = math.sin(rphi)

#         e1   =  cq * dx_g + sq * dy_g   # along-track (dọc)
#         e2   = -sq * dx_g + cq * dy_g   # cross-track (ngang, trong khung xe)
#         e3   = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))  # heading

#         # ── 8. Tính lệnh điều khiển ───────────────────────────────────
#         v_cmd, w_cmd, dbg = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

#         cmd             = Twist()
#         cmd.linear.x    = v_cmd
#         cmd.angular.z   = w_cmd
#         self.cmd_pub.publish(cmd)

#         # ── 9. Giám sát chất lượng ────────────────────────────────────
#         self.sum_cross_sq += cte**2
#         self.log_count    += 1
#         if cte > self.max_cross:
#             self.max_cross = cte
#         rms_cross = math.sqrt(self.sum_cross_sq / self.log_count)

#         if self.log_count % 10 == 0:
#             self.get_logger().info(
#                 f"v={v_cmd:.3f} w={w_cmd:+.3f} | "
#                 f"CTE={cte:.3f} along={nearest_s - self.s:+.3f} | "
#                 f"e1={e1:+.3f} e2={e2:+.3f} e3={math.degrees(e3):+05.1f}deg | "
#                 f"RMS={rms_cross:.4f} max={self.max_cross:.4f}"
#             )

#     def _stop_robot(self):
#         """Phát lệnh dừng hoàn toàn."""
#         stop = Twist()
#         self.cmd_pub.publish(stop)
#         # Phát 3 lần để đảm bảo nhận được (tránh mất gói)
#         for _ in range(3):
#             self.cmd_pub.publish(stop)


# # def main():
# #     rclpy.init()
# #     node = HANav2ExpertNode()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         node.get_logger().info('Ctrl+C → dung robot.')
# #     finally:
# #         node._stop_robot()
# #         node.destroy_node()
# #         rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()
# def main(args=None):
#     rclpy.init(args=args)
    
#     # Khởi tạo node của bạn (Thay HanaV2Node bằng tên class thực tế trong file)
#     node = HANav2ExpertNode() 
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         # Dọn dẹp khi tắt node
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()