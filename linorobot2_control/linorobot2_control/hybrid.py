#!/usr/bin/env python3
"""
hybrid_ha_bs_smc_follower.py
Bộ điều khiển bám quỹ đạo Hybrid: HA (15 Terms) + Backstepping + Adaptive SMC
Tối ưu hóa đặc biệt cho quỹ đạo Zigzag và các góc cua gắt.
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
#  1. HEDGE ALGEBRA ENGINE (Giữ nguyên bản Fixed chuẩn xác)
# ===========================================================================
class HedgeAlgebra15Fixed:
    def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
        self.theta = theta
        self.alpha = alpha
        self.beta  = beta
        self.fm    = {'V': alpha, 'L': beta, 'Small': theta, 'Large': 1.0 - theta}

        self.term_specs = [
            ('Zero',        [],          None   ),      
            ('VVSmall',     ['V', 'V'],  'Small'),      
            ('VSmall',      ['V'],       'Small'),      
            ('IntSmall1',   [],          None   ),      
            ('Small',       [],          'Small'),      
            ('IntSmall2',   [],          None   ),      
            ('VLSmall',     ['V', 'L'],  'Small'),      
            ('LSmall',      ['L'],       'Small'),      
            ('Neutral',     [],          None   ),      
            ('LLarge',      ['L'],       'Large'),      
            ('VLLarge',     ['V', 'L'],  'Large'),      
            ('IntLarge1',   [],          None   ),      
            ('Large',       [],          'Large'),      
            ('VLarge',      ['V'],       'Large'),      
            ('VVLarge',     ['V', 'V'],  'Large'),      
            ('Absolute',    [],          None   ),      
        ]
        self.sem = {}
        self._compute_semantics()
        self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])
        self.hcnt = {
            name: len(hedges) if name not in {'Zero', 'Neutral', 'Absolute', 'Small', 'Large', 'IntSmall1', 'IntSmall2', 'IntLarge1'} else 0
            for name, hedges, _ in self.term_specs
        }

    def _sgn_hedge(self, h): return 1 if h == 'V' else -1
    def _sgn_base(self, h, base): return -1 if (base == 'Small' and h == 'V') or (base == 'Large' and h == 'L') else 1

    def _compute_semantics(self):
        basic_sems = {}
        for name, hedges, base in self.term_specs:
            if name == 'Zero': basic_sems[name] = 0.0
            elif name == 'Neutral': basic_sems[name] = self.theta
            elif name == 'Absolute': basic_sems[name] = 1.0
            elif name not in {'IntSmall1', 'IntSmall2', 'IntLarge1'}:
                fm_base = self.fm[base]
                current_v = (self.theta - self.alpha * fm_base if base == 'Small' else self.theta + self.alpha * fm_base)
                if not hedges: basic_sems[name] = current_v
                else:
                    current_fm = fm_base
                    for h in reversed(hedges):
                        fm_hx = (self.beta if h == 'V' else self.alpha) * current_fm
                        sum_fm = fm_hx if h == 'V' else current_fm
                        sgn_h = self._sgn_hedge(h)
                        sgn_b = self._sgn_base(h, base)
                        current_v += sgn_b * (sum_fm - 0.5 * (1 + sgn_h * (self.beta - self.alpha)) * fm_hx)
                        current_fm = fm_hx
                    basic_sems[name] = current_v
        
        self.sem = basic_sems.copy()
        self.sem['IntSmall1'] = (0.0 + basic_sems['VSmall']) / 2.0  
        self.sem['IntSmall2'] = (basic_sems['Small'] + self.theta) / 2.0  
        self.sem['IntLarge1'] = (self.theta + basic_sems['Large']) / 2.0  

    def fuzzify(self, value):
        v = float(np.clip(value, 0.0, 1.0))
        min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
        cands = [t for t in self.ordered_terms if abs(self.sem[t] - v) <= min_d + 1e-9]
        return min(cands, key=lambda t: self.hcnt.get(t, 3))

    def infer(self, input_val, rule_base):
        term = self.fuzzify(float(np.clip(input_val, 0.0, 1.0)))
        return self.sem[rule_base.get(term, 'Neutral')]

# Rule base được dùng để mapping độ lớn của mặt trượt (Sliding Surface) sang hệ số SMC
def _rule_smc_gain():
    return {
        'Zero':       'Zero',
        'VVSmall':    'VVSmall',
        'VSmall':     'VSmall',
        'IntSmall1':  'Small',
        'Small':      'Small',
        'IntSmall2':  'VLSmall',
        'VLSmall':    'VLSmall',
        'LSmall':     'Neutral',
        'Neutral':    'LLarge',      
        'LLarge':     'VLLarge',
        'VLLarge':    'Large',
        'IntLarge1':  'VLarge',
        'Large':      'VVLarge',
        'VLarge':     'Absolute',
        'VVLarge':    'Absolute',
        'Absolute':   'Absolute',
    }


# ===========================================================================
#  2. HYBRID CONTROLLER: BACKSTEPPING + SMC + HA
# ===========================================================================
class HybridBacksteppingSMC:
    def __init__(self, ha, v_max=0.3, w_max=1.5, dv_max=0.8, dw_max=5.5, Ts=0.05):
        self.ha = ha
        self.v_max = v_max
        self.w_max = w_max
        self.dv_max = dv_max
        self.dw_max = dw_max
        self.Ts = Ts
        
        # --- Thông số Backstepping (Định tuyến cơ sở) ---
        self.c1 = 1.5   # Gain kéo e1 về 0
        self.c2 = 2.5   # Gain kéo e2 về 0 (phụ thuộc v_ref)
        self.c3 = 2.0   # Gain kéo e3 về 0
        
        # --- Thông số Sliding Mode Control ---
        self.lambda_lat = 2.0   # Độ dốc mặt trượt s2 = e3 + lambda*e2
        self.phi_v = 0.05       # Lớp biên (Boundary layer) cho vận tốc dài
        self.phi_w = 0.10       # Lớp biên (Boundary layer) cho vận tốc góc
        
        # --- Thông số Adaptive Gain (Khuếch đại tối đa của HA) ---
        self.K_v_max = 0.2      # Lực cắt SMC tối đa cho v
        self.K_w_max = 1.5      # Lực cắt SMC tối đa cho w (góc cua gắt)
        
        self.MAX_S1 = 0.2       # Chuẩn hóa mặt trượt 1
        self.MAX_S2 = math.pi/2 # Chuẩn hóa mặt trượt 2
        
        self.rule_smc = _rule_smc_gain()
        self.v_prev = 0.0
        self.w_prev = 0.0

    def sat(self, s, phi):
        """Hàm bão hòa mềm chống Chattering"""
        return max(-1.0, min(1.0, s / phi))

    def compute(self, v_ref, w_ref, e1, e2, e3):
        # ---------------------------------------------------------
        # TẦNG 1: BACKSTEPPING KINEMATICS
        # Đảm bảo ổn định Lyapunov toàn cục khi không có nhiễu
        # ---------------------------------------------------------
        v_bs = v_ref * math.cos(e3) + self.c1 * e1
        
        # Chống chia cho 0 khi e3 cực nhỏ
        if abs(e3) > 1e-4:
            sin_e3_div_e3 = math.sin(e3) / e3
        else:
            sin_e3_div_e3 = 1.0
            
        w_bs = w_ref + self.c2 * v_ref * e2 * sin_e3_div_e3 + self.c3 * math.sin(e3)

        # ---------------------------------------------------------
        # TẦNG 2: SLIDING SURFACES
        # Gom các sai số vào 2 mặt phẳng toán học s1, s2
        # ---------------------------------------------------------
        s1 = e1
        s2 = e3 + self.lambda_lat * e2

        # ---------------------------------------------------------
        # TẦNG 3: ADAPTIVE SMC SỬ DỤNG HEDGE ALGEBRA
        # HA tự động tính hệ số bù K tùy thuộc độ lớn của mặt trượt
        # ---------------------------------------------------------
        norm_s1 = min(abs(s1) / self.MAX_S1, 1.0)
        norm_s2 = min(abs(s2) / self.MAX_S2, 1.0)

        # K tự động phình to ở góc cua gắt và xẹp xuống ở đường thẳng
        K_adapt_v = self.ha.infer(norm_s1, self.rule_smc) * self.K_v_max
        K_adapt_w = self.ha.infer(norm_s2, self.rule_smc) * self.K_w_max

        # Lực điều khiển trượt với hàm sat() chống nát hộp số/chattering
        v_smc = K_adapt_v * self.sat(s1, self.phi_v)
        w_smc = K_adapt_w * self.sat(s2, self.phi_w)

        # ---------------------------------------------------------
        # TỔNG HỢP VÀ GIỚI HẠN (RATE LIMITER)
        # ---------------------------------------------------------
        v_raw = v_bs + v_smc
        w_raw = w_bs + w_smc

        # In-place rotation cho các đỉnh zigzag cực gắt
        if abs(e3) > math.radians(60.0):
            v_raw = 0.0
            w_raw = math.copysign(self.w_max, e3)

        v_sat = float(np.clip(v_raw, -self.v_max, self.v_max))
        w_sat = float(np.clip(w_raw, -self.w_max, self.w_max))

        dv = float(np.clip(v_sat - self.v_prev, -self.dv_max * self.Ts, self.dv_max * self.Ts))
        dw = float(np.clip(w_sat - self.w_prev, -self.dw_max * self.Ts, self.dw_max * self.Ts))

        v_cmd = self.v_prev + dv
        w_cmd = self.w_prev + dw

        # Cắt deadband vận tốc quá nhỏ để tránh rít motor
        if abs(w_cmd) < 0.01: w_cmd = 0.0

        self.v_prev = v_cmd
        self.w_prev = w_cmd

        dbg = {
            'v_bs': v_bs, 'w_bs': w_bs,
            'v_smc': v_smc, 'w_smc': w_smc,
            's1': s1, 's2': s2,
            'K_w': K_adapt_w
        }
        return v_cmd, w_cmd, dbg

    def reset(self):
        self.v_prev = self.w_prev = 0.0


# ===========================================================================
#  3. NAV2 TRAJECTORY (Giữ nguyên)
# ===========================================================================
class Nav2Trajectory:
    def __init__(self, path_msg: Path):
        self.xs = np.array([p.pose.position.x for p in path_msg.poses])
        self.ys = np.array([p.pose.position.y for p in path_msg.poses])
        if len(self.xs) < 2:
            self.s_arr, self.L_total = np.array([0.0]), 0.0
            return
        dx, dy = np.diff(self.xs), np.diff(self.ys)
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
        k = min(7, len(self.kappas))
        self.kappas = np.convolve(self.kappas, np.ones(k) / k, mode='same')

    def pos(self, s):
        s = np.clip(s, 0.0, self.L_total)
        return float(np.interp(s, self.s_arr, self.xs)), float(np.interp(s, self.s_arr, self.ys))

    def curvature(self, s):
        s = np.clip(s, 0.0, self.L_total)
        return abs(float(np.interp(s, self.s_arr, self.kappas)))

    def ref_kinematics(self, s, v_s):
        s = np.clip(s, 0.0, self.L_total)
        return float(np.interp(s, self.s_arr, self.phis)), v_s, 0.0

    def find_nearest_s(self, rx, ry):
        if self.L_total == 0.0: return 0.0
        return float(self.s_arr[np.argmin(np.hypot(self.xs - rx, self.ys - ry))])

    def get_cross_track_error(self, rx, ry):
        if self.L_total == 0.0: return 0.0, 0.0
        idx = np.argmin(np.hypot(self.xs - rx, self.ys - ry))
        s_near = self.s_arr[idx]
        x_n, y_n = self.pos(s_near)
        return math.hypot(rx - x_n, ry - y_n), s_near


# ===========================================================================
#  4. ROS2 NODE (Cập nhật logic vòng lặp điều khiển)
# ===========================================================================
class HybridNav2Node(Node):
    LA_K = 2.5    
    LA_MIN = 0.20   
    LA_MAX = 0.50   

    def __init__(self):
        super().__init__('hybrid_ha_bs_smc_node')
        self.declare_parameter('Ts', 0.05)
        self.declare_parameter('v_path_max', 0.30)   
        self.declare_parameter('a_path_max', 0.25)

        self.Ts = self.get_parameter('Ts').value
        self.v_path_max = self.get_parameter('v_path_max').value
        self.a_path_max = self.get_parameter('a_path_max').value

        self.curve_beta = 0.15
        self.num_curve_samp = 10
        self.stop_zone = 0.08   

        self.get_logger().info('=== KHỞI ĐỘNG BỘ ĐIỀU KHIỂN HYBRID HA + BS + SMC ===')

        self.traj = None
        self.ha = HedgeAlgebra15Fixed()
        self.ctrl = HybridBacksteppingSMC(
            ha=self.ha, 
            v_max=self.v_path_max, w_max=1.5, dv_max=0.8, dw_max=5.5, Ts=self.Ts
        )

        self.s = 0.0
        self.v_s = 0.01
        self.curr_q = [0.0, 0.0, 0.0]
        self.pose_received = False
        self.path_received = False
        self.s_initialized = False

        self.sum_cross_sq, self.max_cross, self.log_count = 0.0, 0.0, 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        self.timer = self.create_timer(self.Ts, self.control_loop)

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
        self.ctrl.reset()
        self.sum_cross_sq, self.max_cross, self.log_count = 0.0, 0.0, 0
        self.get_logger().info(f'DA NHAN QUY DAO MOI: {self.traj.L_total:.2f}m')

    def _v_safe(self, s):
        if self.traj is None: return self.v_path_max
        la = self._look_ahead()
        vmin = self.v_path_max
        for i in range(self.num_curve_samp + 1):
            kap = self.traj.curvature(s + i / max(self.num_curve_samp, 1) * la)
            v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
            if v_ok < vmin: vmin = v_ok
        return max(vmin, 0.02)

    def control_loop(self):
        try:
            tf_trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            rx, ry = tf_trans.transform.translation.x, tf_trans.transform.translation.y
            q = tf_trans.transform.rotation
            rphi = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y**2 + q.z**2))
            self.curr_q = [rx, ry, rphi]
            self.pose_received = True
        except TransformException:
            return

        if not self.path_received or self.traj is None: return

        rx, ry, rphi = self.curr_q
        if not self.s_initialized:
            self.s = self.traj.find_nearest_s(rx, ry)
            self.s_initialized = True

        v_adapt = min(self._v_safe(self.s), self.ctrl.v_max)
        dist_left = self.traj.L_total - self.s
        dist_brake = self.v_s**2 / (2.0 * self.a_path_max + 1e-9)

        if dist_left <= dist_brake: a_s = -self.a_path_max
        elif self.v_s < v_adapt - 0.005: a_s = self.a_path_max
        elif self.v_s > v_adapt + 0.005: a_s = -self.a_path_max
        else: a_s = 0.0

        self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.ctrl.v_max))
        self.s += self.v_s * self.Ts

        cte, nearest_s = self.traj.get_cross_track_error(rx, ry)
        la = self._look_ahead()
        if self.s > nearest_s + la: self.s = nearest_s + la
        self.s = min(self.s, self.traj.L_total)

        if math.hypot(rx - self.traj.xs[-1], ry - self.traj.ys[-1]) <= self.stop_zone:
            self._stop_robot()
            rms = math.sqrt(self.sum_cross_sq / max(self.log_count, 1))
            self.get_logger().info(f'DEN DICH! RMS_Error={rms:.4f}m | Max_Error={self.max_cross:.4f}m')
            self.path_received = False
            return

        phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s)
        x_ref, y_ref = self.traj.pos(self.s)

        # Tính toán sai số khung tọa độ di động (Moving Frame)
        dx_g, dy_g = x_ref - rx, y_ref - ry
        cq, sq = math.cos(rphi), math.sin(rphi)
        
        e1 =  cq * dx_g + sq * dy_g   
        e2 = -sq * dx_g + cq * dy_g   
        e3 = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))  

        # Đưa vào lõi Hybrid
        v_cmd, w_cmd, dbg = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v_cmd, w_cmd
        self.cmd_pub.publish(cmd)

        self.sum_cross_sq += cte**2
        self.log_count += 1
        if cte > self.max_cross: self.max_cross = cte

        if self.log_count % 10 == 0:
            rms_cross = math.sqrt(self.sum_cross_sq / self.log_count)
            self.get_logger().info(
                f"SMC_Gain_W: {dbg['K_w']:.3f} | s2: {dbg['s2']:.3f} | "
                f"CTE: {cte:.3f} | RMS: {rms_cross:.4f} | Max: {self.max_cross:.4f}"
            )

    def _stop_robot(self):
        stop = Twist()
        for _ in range(3): self.cmd_pub.publish(stop)

def main(args=None):
    rclpy.init(args=args)
    node = HybridNav2Node() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()