#!/usr/bin/env python3

"""
Figure-8 Trajectory Tracking - HA 3 kenh chuan ly thuyet (Tuned for Speed + Tracking)
=========================================================
CHANGELOG v2 (MIT Trajectory Lab tuning):
  - FIX BUG CHÍNH: v_max 0.40 → 0.55 (phải > v_path_max=0.50, tránh robot tụt sau ref)
  - HAController.MAX_E1: 0.065 → 0.040  (chuẩn hóa nhạy hơn cho sai số nhỏ)
  - HAController.MAX_E2: 0.075 → 0.040  (KEY: phải nhỏ hơn để khai thác dải HA đầy đủ)
  - HAController.MAX_E3: pi/2  → pi/3   (heading correction chặt hơn)
  - gain_e2: 0.80 → 1.40  (bù ngang mạnh hơn – nguyên nhân chính gây lệch)
  - gain_e3: 1.20 → 1.80  (bù heading mạnh hơn – chống cắt góc)
  - gain_e1: 0.20 → 0.25  (nhẹ hơn để không dao động dọc)
  - lpf_alpha: 0.50 → 0.25 (giảm trễ pha w_cmd – robot rẽ kịp hơn)
  - w_max:  3.85 → 4.50   (cho phép bẻ lái mạnh hơn khi cần bù)
  - dw_max: 8.00 → 10.00  (gia tốc góc nhanh hơn)
  - dv_max: 1.50 → 2.00   (tăng tốc giảm tốc dọc nhanh hơn)
  - curve_beta: 0.02 → 0.06 (giảm tốc đủ ở đoạn cong, tránh văng ra)
  - look_ahead_dist: 0.40 → 0.50 (nhìn xa hơn, phòng cua sớm hơn)
  - num_curve_samples: 6 → 8 (ước tính curvature chính xác hơn)
  - a_path_max: 0.20 → 0.25 (gia tốc profile nhanh hơn)
"""

import csv
import datetime
import math
from pathlib import Path as FilePath

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rcl_interfaces.msg import (FloatingPointRange, IntegerRange,
                                 ParameterDescriptor, SetParametersResult)
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path


# ===========================================================================
#  HEDGE ALGEBRA
# ===========================================================================
class HedgeAlgebra:

    def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
        self.theta = theta
        self.alpha = alpha
        self.beta  = beta
        self.fm    = {'V': alpha, 'L': beta,
                      'Small': theta, 'Large': 1.0 - theta}
        self.term_specs = [
            ('Zero',     [],        None   ),
            ('VVSmall',  ['V','V'], 'Small'),
            ('VSmall',   ['V'],     'Small'),
            ('LVSmall',  ['L','V'], 'Small'),
            ('Small',    [],        'Small'),
            ('VLSmall',  ['V','L'], 'Small'),
            ('LSmall',   ['L'],     'Small'),
            ('LLSmall',  ['L','L'], 'Small'),
            ('Neutral',  [],        None   ),
            ('LLLarge',  ['L','L'], 'Large'),
            ('LLarge',   ['L'],     'Large'),
            ('VLLarge',  ['V','L'], 'Large'),
            ('Large',    [],        'Large'),
            ('LVLarge',  ['L','V'], 'Large'),
            ('VLarge',   ['V'],     'Large'),
            ('VVLarge',  ['V','V'], 'Large'),
            ('Absolute', [],        None   ),
        ]
        self.sem = {n: self._compute_sem(n, h, b)
                    for n, h, b in self.term_specs}
        self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])
        self.hcnt = {
            name: len(hedges)
            if name not in {'Zero','Neutral','Absolute','Small','Large'} else 0
            for name, hedges, _ in self.term_specs
        }

    def _sgn_hedge(self, h): return 1 if h == 'V' else -1

    def _sgn_base(self, h, base):
        if base == 'Small': return -1 if h == 'V' else 1
        else:               return  1 if h == 'V' else -1

    def _compute_sem(self, name, hedges, base):
        if name == 'Zero':     return 0.0
        if name == 'Neutral':  return self.theta
        if name == 'Absolute': return 1.0
        fm_base   = self.fm[base]
        current_v = (self.theta - self.alpha * fm_base
                     if base == 'Small'
                     else self.theta + self.alpha * fm_base)
        if not hedges:
            return current_v
        current_fm = fm_base
        for h in reversed(hedges):
            fm_hx  = (self.beta if h == 'V' else self.alpha) * current_fm
            sum_fm = fm_hx if h == 'V' else current_fm
            current_v += self._sgn_base(h, base) * (
                sum_fm
                - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha))
                * fm_hx
            )
            current_fm = fm_hx
        return current_v

    def fuzzify(self, value):
        v     = float(np.clip(value, 0.0, 1.0))
        min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
        cands = [t for t in self.ordered_terms
                 if abs(self.sem[t] - v) <= min_d + 1e-9]
        return min(cands, key=lambda t: self.hcnt.get(t, 3))

    def infer(self, input_val, rule_base):
        term = self.fuzzify(float(np.clip(input_val, 0.0, 1.0)))
        return self.sem[rule_base.get(term, 'Neutral')]

    def sem_table_str(self):
        lines = ['  -- HA Semantic Table (th=%.2f a=%.2f b=%.2f) --'
                 % (self.theta, self.alpha, self.beta)]
        for t in self.ordered_terms:
            bar = '#' * int(round(self.sem[t] * 28))
            lines.append(f'  {t:12s} {self.sem[t]:.4f}  {bar}')
        return '\n'.join(lines)


# ===========================================================================
#  RULE BASES
# ===========================================================================
def _rule_longitudinal():
    return {
        'Zero':     'Zero', 'VVSmall':  'Zero', 'VSmall':   'VVSmall',
        'LVSmall':  'VSmall', 'Small':    'VSmall', 'VLSmall':  'Small',
        'LSmall':   'LSmall', 'LLSmall':  'Large', 'Neutral':  'Large',
        'LLLarge':  'VLarge', 'LLarge':   'VLarge', 'VLLarge':  'VVLarge',
        'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
        'VVLarge':  'Absolute', 'Absolute': 'Absolute',
    }

def _rule_lateral():
    return {
        'Zero':     'Zero', 'VVSmall':  'Zero', 'VSmall':   'Zero',
        'LVSmall':  'VVSmall', 'Small':    'VSmall', 'VLSmall':  'VSmall',
        'LSmall':   'Small', 'LLSmall':  'LSmall', 'Neutral':  'Large',
        'LLLarge':  'Large', 'LLarge':   'VLarge', 'VLLarge':  'VLarge',
        'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
        'VVLarge':  'Absolute', 'Absolute': 'Absolute',
    }

def _rule_heading():
    return {
        'Zero':     'Zero', 'VVSmall':  'Zero', 'VSmall':   'VVSmall',
        'LVSmall':  'VSmall', 'Small':    'Small', 'VLSmall':  'Small',
        'LSmall':   'LSmall', 'LLSmall':  'Large', 'Neutral':  'Large',
        'LLLarge':  'VLarge', 'LLarge':   'VLarge', 'VLLarge':  'VVLarge',
        'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
        'VVLarge':  'Absolute', 'Absolute': 'Absolute',
    }


# ===========================================================================
#  HA CONTROLLER
# ===========================================================================
class HAController:
    # TUNED: Thu hẹp dải chuẩn hóa để khai thác đầy đủ dải HA
    # MAX_E2 cũ = 0.075 → e2=0.02m chỉ dùng 26% dải → bù yếu
    # MAX_E2 mới = 0.040 → e2=0.02m dùng 50% dải → bù mạnh gấp đôi
    MAX_E1 = 0.040   # cũ: 0.065  | nhạy hơn với sai số dọc nhỏ
    MAX_E2 = 0.040   # cũ: 0.075  | KEY FIX: nhạy hơn → chống lệch ngang
    MAX_E3 = (math.pi/3)  # cũ: pi/2 | heading correction chặt hơn

    def __init__(self, ha,
                 gain_e1=0.25, gain_e2=1.40, gain_e3=1.80,
                 v_max=0.55, w_max=4.50, dv_max=2.00, dw_max=10.00,
                 Ts=0.05, lpf_alpha=0.25):
        self.ha        = ha
        self.gain_e1   = gain_e1
        self.gain_e2   = gain_e2
        self.gain_e3   = gain_e3
        self.v_max     = v_max
        self.w_max     = w_max
        self.dv_max    = dv_max
        self.dw_max    = dw_max
        self.Ts        = Ts
        self.lpf_alpha = lpf_alpha
        self.rule_long = _rule_longitudinal()
        self.rule_lat  = _rule_lateral()
        self.rule_head = _rule_heading()
        self.v_prev     = 0.0
        self.w_prev     = 0.0
        self.w_lpf_prev = 0.0

    def update_params(self, **kw):
        for k, v in kw.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)

    def _ha_fb(self, err, max_e, rule, gain):
        if abs(err) <= 0.01 * max_e:
            return 0.0
        norm = min(abs(err) / max_e, 1.0)
        mag  = self.ha.infer(norm, rule) * gain
        return math.copysign(mag, err)

    def compute(self, v_ref, w_ref, e1, e2, e3):
        fb1 = self._ha_fb(e1, self.MAX_E1, self.rule_long, self.gain_e1)
        fb2 = self._ha_fb(e2, self.MAX_E2, self.rule_lat,  self.gain_e2)
        fb3 = self._ha_fb(e3, self.MAX_E3, self.rule_head, self.gain_e3)

        v_raw = v_ref * math.cos(e3) + fb1
        w_raw = w_ref + fb2 + fb3

        # lpf_alpha = 0.25 (cũ 0.50): giảm trễ pha góc, robot rẽ kịp hơn
        w_filt = self.lpf_alpha*self.w_lpf_prev + (1-self.lpf_alpha)*w_raw
        self.w_lpf_prev = w_filt

        v_sat = float(np.clip(v_raw,  -self.v_max, self.v_max))
        w_sat = float(np.clip(w_filt, -self.w_max, self.w_max))

        dt  = self.Ts
        dv  = float(np.clip(v_sat-self.v_prev, -self.dv_max*dt, self.dv_max*dt))
        dw  = float(np.clip(w_sat-self.w_prev, -self.dw_max*dt, self.dw_max*dt))

        v_cmd = self.v_prev + dv
        w_cmd = self.w_prev + dw
        self.v_prev = v_cmd
        self.w_prev = w_cmd
        return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2, 'fb3': fb3}

    def reset(self):
        self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


# ===========================================================================
#  FIGURE-8 TRAJECTORY
# ===========================================================================
class Figure8Trajectory:

    def __init__(self, A, off_x, off_y, lut_size=5000):
        self.A      = A
        self.off_x  = off_x
        self.off_y  = off_y
        self.lut_size = lut_size
        self._build_arc_length_lut()

    def _build_arc_length_lut(self):
        """Tạo ánh xạ độ dài cung s [m] -> tham số hình học u [rad]."""
        self._u_lut = np.linspace(0.0, 2.0 * math.pi, self.lut_size + 1)
        dx_du = self.A * np.cos(self._u_lut)
        dy_du = 2.0 * self.A * np.cos(2.0 * self._u_lut)
        speed = np.hypot(dx_du, dy_du)
        du = self._u_lut[1] - self._u_lut[0]
        increments = 0.5 * (speed[:-1] + speed[1:]) * du
        self._s_lut = np.concatenate(([0.0], np.cumsum(increments)))
        self.loop_length = float(self._s_lut[-1])

    def update_geometry(self, A=None, off_x=None, off_y=None):
        rebuild = A is not None and A != self.A
        if A is not None:
            self.A = A
        if off_x is not None:
            self.off_x = off_x
        if off_y is not None:
            self.off_y = off_y
        if rebuild:
            self._build_arc_length_lut()

    def _u_from_s(self, s):
        s_loop = float(s) % self.loop_length
        return float(np.interp(s_loop, self._s_lut, self._u_lut))

    def pos(self, s):
        u = self._u_from_s(s)
        return (self.off_x + self.A*math.sin(u),
                self.off_y + self.A*math.sin(2.0*u))

    def curvature(self, s):
        u, A = self._u_from_s(s), self.A
        dx   =  A * math.cos(u)
        dy   =  2*A * math.cos(2*u)
        ddx  = -A * math.sin(u)
        ddy  = -4*A * math.sin(2*u)
        return abs(dx*ddy - dy*ddx) / ((dx**2+dy**2)**1.5 + 1e-9)

    def ref_kinematics(self, s, v_s, a_s):
        del a_s  # Gia tốc dọc không làm thay đổi hướng tiếp tuyến.
        u, A = self._u_from_s(s), self.A
        dx = A * math.cos(u)
        dy = 2.0 * A * math.cos(2.0*u)
        ddx = -A * math.sin(u)
        ddy = -4.0 * A * math.sin(2.0*u)
        norm = math.hypot(dx, dy) + 1e-9
        phi_ref = math.atan2(dy, dx)
        signed_kappa = (dx*ddy - dy*ddx) / (norm**3)
        v_ref = v_s
        w_ref = signed_kappa * v_s
        return phi_ref, v_ref, w_ref

    def sample_path(self, n_pts=600):
        return [self.pos(i/n_pts*self.loop_length) for i in range(n_pts+1)]


# ===========================================================================
#  MAIN ROS2 NODE
# ===========================================================================
class Figure8HANode(Node):

    def __init__(self):
        super().__init__('ha_figure8')
        self._declare_params()
        p = self._read_params()

        self.traj = Figure8Trajectory(
            A=p['A'], off_x=p['offset_x'], off_y=p['offset_y'])
        self.ha   = HedgeAlgebra()
        self.ctrl = HAController(
            ha=self.ha,
            gain_e1=p['gain_e1'],   gain_e2=p['gain_e2'],
            gain_e3=p['gain_e3'],   v_max=p['v_max'],
            w_max=p['w_max'],       dv_max=p['dv_max'],
            dw_max=p['dw_max'],     Ts=p['Ts'],
            lpf_alpha=p['lpf_alpha'])

        self.Ts              = p['Ts']
        self.off_x           = p['offset_x']
        self.off_y           = p['offset_y']
        self.off_phi         = p['offset_phi']
        self.v_path_max      = p['v_path_max']
        self.a_path_max      = p['a_path_max']
        self.odom_timeout    = p['odom_timeout']
        self.look_ahead_dist = p['look_ahead_dist']
        self.curve_beta      = p['curve_beta']
        self.num_curve_samp  = int(p['num_curve_samples'])
        self.num_loops       = int(p['num_loops'])
        self.s_end           = self.traj.loop_length * self.num_loops
        self.max_path_poses  = int(p['max_path_poses'])

        self.s           = 0.0
        self.v_s         = 0.02
        self.curr_q      = [p['offset_x'], p['offset_y'], p['offset_phi']]
        self.odom_recv   = False
        self.last_odom_t = None
        self.start_time  = None
        self.done        = False

        self.sum_es_sq  = 0.0
        self.count      = 0
        self.last_log_t = -1
        self.data_log   = []
        self.csv_saved  = False
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = FilePath(p['log_directory']).expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_file = log_dir / f'ha_figure8_{ts}.csv'

        qos_latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub      = self.create_publisher(Twist, p['cmd_vel_topic'], 10)
        self.ref_path_pub = self.create_publisher(Path,  '/ref_path',   qos_latch)
        self.act_path_pub = self.create_publisher(Path,  '/robot_path', 10)
        self.odom_sub     = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'

        self.add_on_set_parameters_callback(self._on_param_change)
        self.create_timer(5.0,     self._pub_ref_path)
        self.create_timer(self.Ts, self._control_loop)

        self.get_logger().info(self._startup_banner(p))

    def _odom_cb(self, msg):
        self.last_odom_t = self.get_clock().now()
        ox  = msg.pose.pose.position.x
        oy  = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        phi = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2))
        c  = math.cos(self.off_phi); s_ = math.sin(self.off_phi)
        self.curr_q[0] = ox*c - oy*s_ + self.off_x
        self.curr_q[1] = ox*s_ + oy*c  + self.off_y
        self.curr_q[2] = math.atan2(math.sin(phi+self.off_phi),
                                    math.cos(phi+self.off_phi))
        self.odom_recv = True

    def _v_safe(self, s):
        v_min = self.v_path_max
        n = max(self.num_curve_samp, 1)
        for i in range(n+1):
            kap  = self.traj.curvature(s + i/n*self.look_ahead_dist)
            v_ok = self.v_path_max / (1.0 + self.curve_beta*kap)
            if v_ok < v_min: v_min = v_ok
        return max(v_min, 0.03)

    def _control_loop(self):
        if self.done or not self.odom_recv: return
        now = self.get_clock().now()

        if self.last_odom_t is not None:
            if (now - self.last_odom_t).nanoseconds*1e-9 > self.odom_timeout:
                self.get_logger().warn('Mat /odom! Dung khan cap.', throttle_duration_sec=2.0)
                self._stop(); return

        if self.start_time is None:
            self.start_time = now; return

        t = (now - self.start_time).nanoseconds * 1e-9

        # A. MOTION PROFILING
        v_adapt    = self._v_safe(self.s)
        dist_brake = self.v_s**2 / (2*self.a_path_max + 1e-9)
        dist_left  = self.s_end - self.s

        if dist_left <= dist_brake:    a_s = -self.a_path_max
        elif self.v_s < v_adapt-0.01: a_s =  self.a_path_max
        elif self.v_s > v_adapt+0.01: a_s = -self.a_path_max
        else:                          a_s =  0.0

        self.v_s = float(np.clip(self.v_s+a_s*self.Ts, 0.01, self.v_path_max))
        self.s  += self.v_s * self.Ts

        if self.s >= self.s_end:
            self.done = True; self._stop(); self._save_csv(); return

        # B. FEEDFORWARD
        phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
        x_ref, y_ref = self.traj.pos(self.s)

        # C. SAI SO
        rx, ry, rphi = self.curr_q
        dx_g = x_ref - rx
        dy_g = y_ref - ry
        cq = math.cos(rphi); sq = math.sin(rphi)
        e1 =  cq*dx_g + sq*dy_g
        e2 = -sq*dx_g + cq*dy_g
        e3 = math.atan2(math.sin(phi_ref-rphi), math.cos(phi_ref-rphi))

        # D. HA FEEDBACK
        v_cmd, w_cmd, dbg = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

        # E. PUBLISH
        cmd = Twist()
        cmd.linear.x  = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

        # F. LOG
        es    = math.hypot(dx_g, dy_g)
        kappa = self.traj.curvature(self.s)
        self._log_viz(now, t, es, e1, e2, e3, v_cmd, w_cmd, v_ref, w_ref, v_adapt, kappa, dbg)

    def _log_viz(self, now, t, es, e1, e2, e3, v, w, v_ref, w_ref, v_adp, kap, dbg):
        ps = PoseStamped()
        ps.header.frame_id = 'world'
        ps.header.stamp    = now.to_msg()
        ps.pose.position.x = self.curr_q[0]
        ps.pose.position.y = self.curr_q[1]
        half_yaw = 0.5 * self.curr_q[2]
        ps.pose.orientation.z = math.sin(half_yaw)
        ps.pose.orientation.w = math.cos(half_yaw)
        self.actual_path.poses.append(ps)
        if len(self.actual_path.poses) > self.max_path_poses:
            del self.actual_path.poses[:-self.max_path_poses]
        self.actual_path.header.stamp = now.to_msg()
        self.act_path_pub.publish(self.actual_path)

        self.sum_es_sq += es**2
        self.count     += 1
        rms = math.sqrt(self.sum_es_sq / self.count)

        self.data_log.append([
            round(t, 3), round(es, 5), round(rms, 5),
            round(e1, 5), round(e2, 5), round(e3, 5),
            round(v, 5),  round(w, 5),
            round(v_ref, 5), round(w_ref, 5),
            round(v_adp, 5), round(kap, 4),
            round(dbg['fb1'], 5), round(dbg['fb2'], 5), round(dbg['fb3'], 5),
        ])

        if int(t) > self.last_log_t:
            self.get_logger().info(f't={t:6.1f}s | es={es:.4f}m RMSE={rms:.4f}m | v={v:.4f} w={w:.4f}')
            self.last_log_t = int(t)

    def _pub_ref_path(self):
        path = Path()
        path.header.frame_id = 'world'
        path.header.stamp    = self.get_clock().now().to_msg()
        for px, py in self.traj.sample_path(600):
            ps = PoseStamped()
            ps.header.frame_id = 'world'
            ps.pose.position.x = px
            ps.pose.position.y = py
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.ref_path_pub.publish(path)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info('=== ROBOT DUNG ===', once=True)

    def _save_csv(self):
        if self.csv_saved or not self.data_log:
            return
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time(s)', 'es(m)', 'RMSE(m)', 'e1_long(m)', 'e2_lat(m)', 'e3_head(rad)',
                             'v_cmd(m/s)', 'w_cmd(rad/s)', 'v_ref(m/s)', 'w_ref(rad/s)',
                             'v_adaptive(m/s)', 'kappa(1/m)', 'fb1_v(m/s)', 'fb2_w(rad/s)', 'fb3_w(rad/s)'])
            writer.writerows(self.data_log)
        self.csv_saved = True
        self.get_logger().info(f'Saved CSV: {self.csv_file}')

    def _declare_params(self):
        def fp(desc, lo, hi):
            return ParameterDescriptor(
                description=desc,
                floating_point_range=[FloatingPointRange(
                    from_value=float(lo), to_value=float(hi), step=0.0)])

        def ip(desc, lo, hi):
            return ParameterDescriptor(
                description=desc,
                integer_range=[IntegerRange(
                    from_value=int(lo), to_value=int(hi), step=1)])

        self.declare_parameter('A',           0.70, fp('Bien do [m]',    0.10, 3.0))
        self.declare_parameter('num_loops',   5,    ip('So vong', 1, 1000))
        self.declare_parameter('offset_x',    1.10, fp('Tam X [m]',     -5.0, 5.0))
        self.declare_parameter('offset_y',    0.90, fp('Tam Y [m]',     -5.0, 5.0))
        self.declare_parameter('offset_phi',  0.00, fp('Huong [rad]',   -math.pi, math.pi))

        # v_path_max giữ nguyên 0.50, nhưng v_max PHẢI > v_path_max (fix bug chính)
        self.declare_parameter('v_path_max',  0.50, fp('Vmax tho [m/s]', 0.02, 2.00))
        self.declare_parameter('a_path_max',  0.25, fp('Gia toc [m/s2]', 0.01, 2.00))  # cũ 0.20
        self.declare_parameter('Ts',          0.05, fp('Chu ky [s]',     0.01, 0.20))

        # Gain tăng mạnh để bù sai số kịp thời ở tốc độ cao
        self.declare_parameter('gain_e1',     0.25, fp('Gain e1 [m/s]',  0.00, 1.00))  # cũ 0.20
        self.declare_parameter('gain_e2',     1.40, fp('Gain e2 [rad/s]',0.00, 3.00))  # cũ 0.80 ← KEY
        self.declare_parameter('gain_e3',     1.80, fp('Gain e3 [rad/s]',0.00, 3.00))  # cũ 1.20 ← KEY

        # v_max PHẢI > v_path_max=0.50 → dùng 0.55 (FIX BUG CHÍNH)
        self.declare_parameter('v_max',       0.55, fp('Vmax robot',     0.05, 2.00))  # cũ 0.40 ← BUG FIX
        self.declare_parameter('w_max',       4.50, fp('Wmax [rad/s]',   0.10, 8.00))  # cũ 3.85
        self.declare_parameter('dv_max',      2.00, fp('Rate v',         0.10, 5.00))  # cũ 1.50
        self.declare_parameter('dw_max',     10.00, fp('Rate w',         0.10, 15.00)) # cũ 8.00

        # lpf_alpha nhỏ hơn = phản hồi góc nhanh hơn = giảm trễ pha
        self.declare_parameter('lpf_alpha',   0.25, fp('LPF alpha',      0.00, 0.95))  # cũ 0.50 ← KEY
        self.declare_parameter('odom_timeout',0.50, fp('Timeout odom',   0.10, 5.00))
        self.declare_parameter('look_ahead_dist', 0.50, fp('Look-ahead', 0.10, 2.00))  # cũ 0.40
        self.declare_parameter('curve_beta',      0.06, fp('Beta cua',   0.00, 0.20))  # cũ 0.02 ← KEY
        self.declare_parameter('num_curve_samples', 8, ip('So mau k', 1, 1000))
        self.declare_parameter('max_path_poses', 2000, ip('So pose toi da cua robot_path', 10, 100000))
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('log_directory', '.')

    def _read_params(self):
        names = ['A','num_loops','offset_x','offset_y','offset_phi',
                 'v_path_max','a_path_max','Ts',
                 'gain_e1','gain_e2','gain_e3',
                 'v_max','w_max','dv_max','dw_max','lpf_alpha',
                 'odom_timeout','look_ahead_dist','curve_beta','num_curve_samples',
                 'max_path_poses','cmd_vel_topic','log_directory']
        return {n: self.get_parameter(n).value for n in names}

    def _on_param_change(self, params):
        cur = self._read_params()
        new = {p.name: p.value for p in params}
        read_only = {'cmd_vel_topic', 'log_directory'}
        changed_read_only = read_only.intersection(new)
        if changed_read_only:
            names = ', '.join(sorted(changed_read_only))
            return SetParametersResult(
                successful=False, reason=f'{names} chi thay doi khi khoi dong node.')
        if 'Ts' in new: return SetParametersResult(successful=False, reason='Ts chi doc.')
        if new.get('v_path_max', cur['v_path_max']) > new.get('v_max', cur['v_max']):
            return SetParametersResult(successful=False, reason='v_path_max phai <= v_max')
        self.ctrl.update_params(
            gain_e1=new.get('gain_e1', cur['gain_e1']),
            gain_e2=new.get('gain_e2', cur['gain_e2']),
            gain_e3=new.get('gain_e3', cur['gain_e3']),
            v_max=new.get('v_max', cur['v_max']),
            w_max=new.get('w_max', cur['w_max']),
            dv_max=new.get('dv_max', cur['dv_max']),
            dw_max=new.get('dw_max', cur['dw_max']),
            lpf_alpha=new.get('lpf_alpha', cur['lpf_alpha']),
        )
        for attr in ('v_path_max','a_path_max','odom_timeout','look_ahead_dist','curve_beta'):
            if attr in new: setattr(self, attr, new[attr])
        if 'num_curve_samples' in new:
            self.num_curve_samp = int(new['num_curve_samples'])
        if 'max_path_poses' in new:
            self.max_path_poses = int(new['max_path_poses'])

        geometry_changed = any(k in new for k in ('A', 'offset_x', 'offset_y'))
        if geometry_changed:
            old_progress = self.s / max(self.traj.loop_length, 1e-9)
            self.traj.update_geometry(
                A=new.get('A'),
                off_x=new.get('offset_x'),
                off_y=new.get('offset_y'))
            self.off_x = new.get('offset_x', self.off_x)
            self.off_y = new.get('offset_y', self.off_y)
            self.s = old_progress * self.traj.loop_length

        if 'offset_phi' in new:
            self.off_phi = new['offset_phi']
        if 'num_loops' in new:
            self.num_loops = int(new['num_loops'])
        if geometry_changed or 'num_loops' in new:
            self.s_end = self.traj.loop_length * self.num_loops

        return SetParametersResult(successful=True)

    def _startup_banner(self, p):
        return (f"HA Figure-8 v2 (MIT-tuned) | v_path_max={p['v_path_max']} "
                f"v_max={p['v_max']} | gain_e2={p['gain_e2']} gain_e3={p['gain_e3']} "
                f"lpf={p['lpf_alpha']} curve_beta={p['curve_beta']}")


# ===========================================================================
def main():
    rclpy.init()
    node = Figure8HANode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node._stop()
        node._save_csv()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# """
# Figure-8 Trajectory Tracking - HA 3 kenh chuan ly thuyet (MIT Tuned)
# =========================================================
# """
# import csv
# import datetime
# import math

# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.parameter import Parameter
# from rclpy.qos import DurabilityPolicy, QoSProfile
# from rcl_interfaces.msg import (FloatingPointRange, ParameterDescriptor,
#                                  SetParametersResult)
# from geometry_msgs.msg import PoseStamped, Twist
# from nav_msgs.msg import Odometry, Path


# # ===========================================================================
# #  HEDGE ALGEBRA
# # ===========================================================================
# class HedgeAlgebra:
#     def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
#         self.theta = theta
#         self.alpha = alpha
#         self.beta  = beta
#         self.fm    = {'V': alpha, 'L': beta,
#                       'Small': theta, 'Large': 1.0 - theta}
#         self.term_specs = [
#             ('Zero',     [],        None   ),
#             ('VVSmall',  ['V','V'], 'Small'),
#             ('VSmall',   ['V'],     'Small'),
#             ('LVSmall',  ['L','V'], 'Small'),
#             ('Small',    [],        'Small'),
#             ('VLSmall',  ['V','L'], 'Small'),
#             ('LSmall',   ['L'],     'Small'),
#             ('LLSmall',  ['L','L'], 'Small'),
#             ('Neutral',  [],        None   ),
#             ('LLLarge',  ['L','L'], 'Large'),
#             ('LLarge',   ['L'],     'Large'),
#             ('VLLarge',  ['V','L'], 'Large'),
#             ('Large',    [],        'Large'),
#             ('LVLarge',  ['L','V'], 'Large'),
#             ('VLarge',   ['V'],     'Large'),
#             ('VVLarge',  ['V','V'], 'Large'),
#             ('Absolute', [],        None   ),
#         ]
#         self.sem = {n: self._compute_sem(n, h, b)
#                     for n, h, b in self.term_specs}
#         self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])
#         self.hcnt = {
#             name: len(hedges)
#             if name not in {'Zero','Neutral','Absolute','Small','Large'} else 0
#             for name, hedges, _ in self.term_specs
#         }

#     def _sgn_hedge(self, h): return 1 if h == 'V' else -1

#     def _sgn_base(self, h, base):
#         if base == 'Small': return -1 if h == 'V' else 1
#         else:               return  1 if h == 'V' else -1

#     def _compute_sem(self, name, hedges, base):
#         if name == 'Zero':     return 0.0
#         if name == 'Neutral':  return self.theta
#         if name == 'Absolute': return 1.0
#         fm_base   = self.fm[base]
#         current_v = (self.theta - self.alpha * fm_base
#                      if base == 'Small'
#                      else self.theta + self.alpha * fm_base)
#         if not hedges:
#             return current_v
#         current_fm = fm_base
#         for h in reversed(hedges):
#             fm_hx  = (self.beta if h == 'V' else self.alpha) * current_fm
#             sum_fm = fm_hx if h == 'V' else current_fm
#             current_v += self._sgn_base(h, base) * (
#                 sum_fm
#                 - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha))
#                 * fm_hx
#             )
#             current_fm = fm_hx
#         return current_v

#     def fuzzify(self, value):
#         v     = float(np.clip(value, 0.0, 1.0))
#         min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
#         cands = [t for t in self.ordered_terms
#                  if abs(self.sem[t] - v) <= min_d + 1e-9]
#         return min(cands, key=lambda t: self.hcnt.get(t, 3))

#     def infer(self, input_val, rule_base):
#         term = self.fuzzify(float(np.clip(input_val, 0.0, 1.0)))
#         return self.sem[rule_base.get(term, 'Neutral')]

#     def sem_table_str(self):
#         lines = ['  -- HA Semantic Table (th=%.2f a=%.2f b=%.2f) --'
#                  % (self.theta, self.alpha, self.beta)]
#         for t in self.ordered_terms:
#             bar = '#' * int(round(self.sem[t] * 28))
#             lines.append(f'  {t:12s} {self.sem[t]:.4f}  {bar}')
#         return '\n'.join(lines)


# # ===========================================================================
# #  RULE BASES
# # ===========================================================================
# def _rule_longitudinal():
#     return {
#         'Zero':     'Zero', 'VVSmall':  'Zero', 'VSmall':   'VVSmall',
#         'LVSmall':  'VSmall', 'Small':    'VSmall', 'VLSmall':  'Small',
#         'LSmall':   'LSmall', 'LLSmall':  'Large', 'Neutral':  'Large',
#         'LLLarge':  'VLarge', 'LLarge':   'VLarge', 'VLLarge':  'VVLarge',
#         'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
#         'VVLarge':  'Absolute', 'Absolute': 'Absolute',
#     }

# def _rule_lateral():
#     return {
#         'Zero':     'Zero', 'VVSmall':  'Zero', 'VSmall':   'Zero',
#         'LVSmall':  'VVSmall', 'Small':    'VSmall', 'VLSmall':  'VSmall',
#         'LSmall':   'Small', 'LLSmall':  'LSmall', 'Neutral':  'Large',
#         'LLLarge':  'Large', 'LLarge':   'VLarge', 'VLLarge':  'VLarge',
#         'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
#         'VVLarge':  'Absolute', 'Absolute': 'Absolute',
#     }

# def _rule_heading():
#     return {
#         'Zero':     'Zero', 'VVSmall':  'Zero', 'VSmall':   'VVSmall',
#         'LVSmall':  'VSmall', 'Small':    'Small', 'VLSmall':  'Small',
#         'LSmall':   'LSmall', 'LLSmall':  'Large', 'Neutral':  'Large',
#         'LLLarge':  'VLarge', 'LLarge':   'VLarge', 'VLLarge':  'VVLarge',
#         'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
#         'VVLarge':  'Absolute', 'Absolute': 'Absolute',
#     }


# # ===========================================================================
# #  HA CONTROLLER
# # ===========================================================================
# class HAController:
#     MAX_E1 = 0.065   
#     MAX_E2 = 0.075   
#     MAX_E3 = (math.pi/2) 

#     def __init__(self, ha,
#                  gain_e1=0.25, gain_e2=1.50, gain_e3=2.00,
#                  v_max=0.40, w_max=3.85, dv_max=1.50, dw_max=8.00,
#                  Ts=0.05, lpf_alpha=0.30):
#         self.ha        = ha
#         self.gain_e1   = gain_e1
#         self.gain_e2   = gain_e2
#         self.gain_e3   = gain_e3
#         self.v_max     = v_max
#         self.w_max     = w_max
#         self.dv_max    = dv_max
#         self.dw_max    = dw_max
#         self.Ts        = Ts
#         self.lpf_alpha = lpf_alpha
#         self.rule_long = _rule_longitudinal()
#         self.rule_lat  = _rule_lateral()
#         self.rule_head = _rule_heading()
#         self.v_prev     = 0.0
#         self.w_prev     = 0.0
#         self.w_lpf_prev = 0.0

#     def update_params(self, **kw):
#         for k, v in kw.items():
#             if v is not None and hasattr(self, k):
#                 setattr(self, k, v)

#     def _ha_fb(self, err, max_e, rule, gain):
#         if abs(err) <= 0.01 * max_e:
#             return 0.0
#         norm = min(abs(err) / max_e, 1.0)
#         mag  = self.ha.infer(norm, rule) * gain
#         return math.copysign(mag, err)

#     def compute(self, v_ref, w_ref, e1, e2, e3):
#         fb1 = self._ha_fb(e1, self.MAX_E1, self.rule_long, self.gain_e1)
#         fb2 = self._ha_fb(e2, self.MAX_E2, self.rule_lat,  self.gain_e2)
#         fb3 = self._ha_fb(e3, self.MAX_E3, self.rule_head, self.gain_e3)

#         v_raw = v_ref * math.cos(e3) + fb1
#         w_raw = w_ref + fb2 + fb3

#         w_filt = self.lpf_alpha*self.w_lpf_prev + (1-self.lpf_alpha)*w_raw
#         self.w_lpf_prev = w_filt

#         v_sat = float(np.clip(v_raw,  -self.v_max, self.v_max))
#         w_sat = float(np.clip(w_filt, -self.w_max, self.w_max))

#         dt  = self.Ts
#         dv  = float(np.clip(v_sat-self.v_prev, -self.dv_max*dt, self.dv_max*dt))
#         dw  = float(np.clip(w_sat-self.w_prev, -self.dw_max*dt, self.dw_max*dt))

#         v_cmd = self.v_prev + dv
#         w_cmd = self.w_prev + dw
#         self.v_prev = v_cmd
#         self.w_prev = w_cmd
#         return v_cmd, w_cmd, {'fb1': fb1, 'fb2': fb2, 'fb3': fb3}

#     def reset(self):
#         self.v_prev = self.w_prev = self.w_lpf_prev = 0.0


# # ===========================================================================
# #  FIGURE-8 TRAJECTORY
# # ===========================================================================
# class Figure8Trajectory:
#     def __init__(self, A, off_x, off_y, L_loop=30.0):
#         self.A      = A
#         self.off_x  = off_x
#         self.off_y  = off_y
#         self.L_loop = L_loop
#         self.freq   = 2.0 * math.pi / L_loop

#     def pos(self, s):
#         f = self.freq
#         return (self.off_x + self.A*math.sin(f*s),
#                 self.off_y + self.A*math.sin(2*f*s))

#     def curvature(self, s):
#         f, A = self.freq, self.A
#         dx   =  A*f    * math.cos(f*s)
#         dy   =  2*A*f  * math.cos(2*f*s)
#         ddx  = -A*f**2 * math.sin(f*s)
#         ddy  = -4*A*f**2 * math.sin(2*f*s)
#         return abs(dx*ddy - dy*ddx) / ((dx**2+dy**2)**1.5 + 1e-9)

#     def ref_kinematics(self, s, v_s, a_s):
#         f, A   = self.freq, self.A
#         dx_ds  =  A*f    * math.cos(f*s)
#         dy_ds  =  2*A*f  * math.cos(2*f*s)
#         ddx_ds = -A*f**2 * math.sin(f*s)
#         ddy_ds = -4*A*f**2 * math.sin(2*f*s)
#         dx_dt  = dx_ds*v_s;  dy_dt  = dy_ds*v_s
#         ddx_dt = ddx_ds*v_s**2 + dx_ds*a_s
#         ddy_dt = ddy_ds*v_s**2 + dy_ds*a_s
#         phi_ref = math.atan2(dy_dt, dx_dt)
#         v_ref   = math.hypot(dx_dt, dy_dt)
#         denom   = dx_dt**2 + dy_dt**2 + 1e-9
#         w_ref   = (dx_dt*ddy_dt - dy_dt*ddx_dt) / denom
#         return phi_ref, v_ref, w_ref

#     def sample_path(self, n_pts=600):
#         return [self.pos(i/n_pts*self.L_loop) for i in range(n_pts+1)]


# # ===========================================================================
# #  MAIN ROS2 NODE
# # ===========================================================================
# class Figure8HANode(Node):
#     L_LOOP = 30.0

#     def __init__(self):
#         super().__init__(
#             'ha_figure8',
#             parameter_overrides=[
#                 Parameter('use_sim_time', Parameter.Type.BOOL, True)
#             ]
#         )
#         self._declare_params()
#         p = self._read_params()

#         self.traj = Figure8Trajectory(
#             A=p['A'], off_x=p['offset_x'], off_y=p['offset_y'],
#             L_loop=self.L_LOOP)
#         self.ha   = HedgeAlgebra()
#         self.ctrl = HAController(
#             ha=self.ha,
#             gain_e1=p['gain_e1'],   gain_e2=p['gain_e2'],
#             gain_e3=p['gain_e3'],   v_max=p['v_max'],
#             w_max=p['w_max'],       dv_max=p['dv_max'],
#             dw_max=p['dw_max'],     Ts=p['Ts'],
#             lpf_alpha=p['lpf_alpha'])

#         self.Ts              = p['Ts']
#         self.off_x           = p['offset_x']
#         self.off_y           = p['offset_y']
#         self.off_phi         = p['offset_phi']
#         self.v_path_max      = p['v_path_max']
#         self.a_path_max      = p['a_path_max']
#         self.odom_timeout    = p['odom_timeout']
#         self.look_ahead_dist = p['look_ahead_dist']
#         self.curve_beta      = p['curve_beta']
#         self.num_curve_samp  = int(p['num_curve_samples'])
#         self.s_end           = self.L_LOOP * p['num_loops']

#         self.s           = 0.0
#         self.v_s         = 0.02
#         self.curr_q      = [p['offset_x'], p['offset_y'], p['offset_phi']]
#         self.odom_recv   = False
#         self.last_odom_t = None
#         self.start_time  = None
#         self.done        = False

#         self.sum_es_sq  = 0.0
#         self.count      = 0
#         self.last_log_t = -1
#         self.data_log   = []
#         ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.csv_file = f'ha_figure8_{ts}.csv'

#         qos_latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
#         self.cmd_pub      = self.create_publisher(Twist, '/cmd_vel',    10)
#         self.ref_path_pub = self.create_publisher(Path,  '/ref_path',   qos_latch)
#         self.act_path_pub = self.create_publisher(Path,  '/robot_path', 10)
#         self.odom_sub     = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

#         self.actual_path = Path()
#         self.actual_path.header.frame_id = 'world'

#         self.add_on_set_parameters_callback(self._on_param_change)
#         self.create_timer(5.0,     self._pub_ref_path)
#         self.create_timer(self.Ts, self._control_loop)

#         self.get_logger().info(self._startup_banner(p))

#     def _odom_cb(self, msg):
#         self.last_odom_t = self.get_clock().now()
#         ox  = msg.pose.pose.position.x
#         oy  = msg.pose.pose.position.y
#         q   = msg.pose.pose.orientation
#         phi = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y**2+q.z**2))
#         c  = math.cos(self.off_phi); s_ = math.sin(self.off_phi)
#         self.curr_q[0] = ox*c - oy*s_ + self.off_x
#         self.curr_q[1] = ox*s_ + oy*c  + self.off_y
#         self.curr_q[2] = math.atan2(math.sin(phi+self.off_phi),
#                                     math.cos(phi+self.off_phi))
#         self.odom_recv = True

#     def _v_safe(self, s):
#         v_min = self.v_path_max
#         n = max(self.num_curve_samp, 1)
#         for i in range(n+1):
#             kap  = self.traj.curvature(s + i/n*self.look_ahead_dist)
#             v_ok = self.v_path_max / (1.0 + self.curve_beta*kap)
#             if v_ok < v_min: v_min = v_ok
#         return max(v_min, 0.03)

#     def _control_loop(self):
#         if self.done or not self.odom_recv: return
#         now = self.get_clock().now()

#         if self.last_odom_t is not None:
#             if (now - self.last_odom_t).nanoseconds*1e-9 > self.odom_timeout:
#                 self.get_logger().warn('Mat /odom! Dung khan cap.', throttle_duration_sec=2.0)
#                 self._stop(); return

#         if self.start_time is None:
#             self.start_time = now; return

#         t = (now - self.start_time).nanoseconds * 1e-9

#         # A. MOTION PROFILING
#         v_adapt    = self._v_safe(self.s)
#         dist_brake = self.v_s**2 / (2*self.a_path_max + 1e-9)
#         dist_left  = self.s_end - self.s

#         if dist_left <= dist_brake:    a_s = -self.a_path_max
#         elif self.v_s < v_adapt-0.01: a_s =  self.a_path_max
#         elif self.v_s > v_adapt+0.01: a_s = -self.a_path_max
#         else:                          a_s =  0.0

#         self.v_s = float(np.clip(self.v_s+a_s*self.Ts, 0.01, self.v_path_max))
#         self.s  += self.v_s * self.Ts

#         if self.s >= self.s_end:
#             self.done = True; self._stop(); self._save_csv(); return

#         # B. FEEDFORWARD
#         phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
#         x_ref, y_ref = self.traj.pos(self.s)

#         # C. SAI SO
#         rx, ry, rphi = self.curr_q
#         dx_g = x_ref - rx
#         dy_g = y_ref - ry
#         cq = math.cos(rphi); sq = math.sin(rphi)
#         e1 =  cq*dx_g + sq*dy_g    
#         e2 = -sq*dx_g + cq*dy_g    
#         e3 = math.atan2(math.sin(phi_ref-rphi), math.cos(phi_ref-rphi))

#         # D. HA FEEDBACK
#         v_cmd, w_cmd, dbg = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

#         # E. PUBLISH
#         cmd = Twist()
#         cmd.linear.x  = v_cmd
#         cmd.angular.z = w_cmd
#         self.cmd_pub.publish(cmd)

#         # F. LOG
#         es    = math.hypot(dx_g, dy_g)
#         kappa = self.traj.curvature(self.s)
#         self._log_viz(now, t, es, e1, e2, e3, v_cmd, w_cmd, v_ref, w_ref, v_adapt, kappa, dbg)

#     def _log_viz(self, now, t, es, e1, e2, e3, v, w, v_ref, w_ref, v_adp, kap, dbg):
#         ps = PoseStamped()
#         ps.header.frame_id = 'world'
#         ps.header.stamp    = now.to_msg()
#         ps.pose.position.x = self.curr_q[0]
#         ps.pose.position.y = self.curr_q[1]
#         self.actual_path.poses.append(ps)
#         self.actual_path.header.stamp = now.to_msg()
#         self.act_path_pub.publish(self.actual_path)

#         self.sum_es_sq += es**2
#         self.count     += 1
#         rms = math.sqrt(self.sum_es_sq / self.count)

#         self.data_log.append([
#             round(t, 3), round(es, 5), round(rms, 5),
#             round(e1, 5), round(e2, 5), round(e3, 5),
#             round(v, 5),  round(w, 5),
#             round(v_ref, 5), round(w_ref, 5),
#             round(v_adp, 5), round(kap, 4),
#             round(dbg['fb1'], 5), round(dbg['fb2'], 5), round(dbg['fb3'], 5),
#         ])

#         if int(t) > self.last_log_t:
#             self.get_logger().info(f't={t:6.1f}s | es={es:.4f}m RMSE={rms:.4f}m | v={v:.4f} w={w:.4f}')
#             self.last_log_t = int(t)

#     def _pub_ref_path(self):
#         path = Path()
#         path.header.frame_id = 'world'
#         path.header.stamp    = self.get_clock().now().to_msg()
#         for px, py in self.traj.sample_path(600):
#             ps = PoseStamped()
#             ps.header.frame_id = 'world'
#             ps.pose.position.x = px
#             ps.pose.position.y = py
#             path.poses.append(ps)
#         self.ref_path_pub.publish(path)

#     def _stop(self):
#         self.cmd_pub.publish(Twist())
#         self.get_logger().info('=== ROBOT DUNG ===', once=True)

#     def _save_csv(self):
#         if not self.data_log: return
#         with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow(['time(s)', 'es(m)', 'RMSE(m)', 'e1_long(m)', 'e2_lat(m)', 'e3_head(rad)',
#                              'v_cmd(m/s)', 'w_cmd(rad/s)', 'v_ref(m/s)', 'w_ref(rad/s)',
#                              'v_adaptive(m/s)', 'kappa(1/m)', 'fb1_v(m/s)', 'fb2_w(rad/s)', 'fb3_w(rad/s)'])
#             writer.writerows(self.data_log)
#         self.get_logger().info('Saved CSV.')

#     # ----------------------------------------------------------------
#     # THAY ĐỔI THÔNG SỐ: Giảm tốc độ trần, Tăng rà phanh góc cua, Siết Gains
#     # ----------------------------------------------------------------
#     def _declare_params(self):
#         def fp(desc, lo, hi):
#             return ParameterDescriptor(
#                 description=desc,
#                 floating_point_range=[FloatingPointRange(
#                     from_value=float(lo), to_value=float(hi), step=0.0)])
        
#         self.declare_parameter('A',           0.70, fp('Bien do [m]',    0.10, 3.0))
#         self.declare_parameter('num_loops',   5,    ParameterDescriptor(description='So vong'))
#         self.declare_parameter('offset_x',    1.10, fp('Tam X [m]',     -5.0, 5.0))
#         self.declare_parameter('offset_y',    0.90, fp('Tam Y [m]',     -5.0, 5.0))
#         self.declare_parameter('offset_phi',  0.00, fp('Huong [rad]',   -math.pi, math.pi))
        
#         # Hạ trần vận tốc xuống 0.35m/s (khoảng 1 phút 25s - 1 phút 30s / vòng)
#         self.declare_parameter('v_path_max',  0.35, fp('Vmax tho [m/s]', 0.02, 2.00))
#         self.declare_parameter('a_path_max',  0.20, fp('Gia toc [m/s2]', 0.01, 2.00))
#         self.declare_parameter('Ts',          0.05, fp('Chu ky [s]',     0.01, 0.20))
        
#         # Siết chặt Gain: Tăng gắt ở trục ngang (e2) và hướng (e3) để ép xe về quỹ đạo
#         self.declare_parameter('gain_e1',     0.25, fp('Gain e1 [m/s]',  0.00, 1.00))
#         self.declare_parameter('gain_e2',     1.50, fp('Gain e2 [rad/s]',0.00, 3.00))
#         self.declare_parameter('gain_e3',     2.00, fp('Gain e3 [rad/s]',0.00, 3.00))
        
#         self.declare_parameter('v_max',       0.40, fp('Vmax robot',     0.05, 2.00))
#         self.declare_parameter('w_max',       3.85, fp('Wmax [rad/s]',   0.10, 8.00))
#         self.declare_parameter('dv_max',      1.50, fp('Rate v',         0.10, 5.00))
#         self.declare_parameter('dw_max',      8.00, fp('Rate w',         0.10, 15.00))
        
#         # Giảm LPF alpha để tín hiệu lái nhạy hơn, không bị trễ
#         self.declare_parameter('lpf_alpha',   0.30, fp('LPF alpha',      0.00, 0.95))
#         self.declare_parameter('odom_timeout',0.50, fp('Timeout odom',   0.10, 5.00))
        
#         # TĂNG ĐỘ THÔNG MINH KHI ÔM CUA: Nhìn xa hơn, giảm tốc gắt hơn khi thấy góc cong
#         self.declare_parameter('look_ahead_dist', 0.60, fp('Look-ahead', 0.10, 2.00))
#         self.declare_parameter('curve_beta',      0.40, fp('Beta cua',   0.00, 1.00))
#         self.declare_parameter('num_curve_samples', 6, ParameterDescriptor(description='So mau k'))

#     def _read_params(self):
#         names = ['A','num_loops','offset_x','offset_y','offset_phi',
#                  'v_path_max','a_path_max','Ts',
#                  'gain_e1','gain_e2','gain_e3',
#                  'v_max','w_max','dv_max','dw_max','lpf_alpha',
#                  'odom_timeout','look_ahead_dist','curve_beta','num_curve_samples']
#         return {n: self.get_parameter(n).value for n in names}

#     def _on_param_change(self, params):
#         cur = self._read_params()
#         new = {p.name: p.value for p in params}
#         if 'Ts' in new: return SetParametersResult(successful=False, reason='Ts chi doc.')
#         if new.get('v_path_max', cur['v_path_max']) > new.get('v_max', cur['v_max']):
#             return SetParametersResult(successful=False, reason='v_path_max phai <= v_max')
#         self.ctrl.update_params(
#             gain_e1=new.get('gain_e1', cur['gain_e1']),
#             gain_e2=new.get('gain_e2', cur['gain_e2']),
#             gain_e3=new.get('gain_e3', cur['gain_e3']),
#             v_max=new.get('v_max', cur['v_max']),
#             w_max=new.get('w_max', cur['w_max']),
#             dv_max=new.get('dv_max', cur['dv_max']),
#             dw_max=new.get('dw_max', cur['dw_max']),
#             lpf_alpha=new.get('lpf_alpha', cur['lpf_alpha']),
#         )
#         for attr in ('v_path_max','a_path_max','odom_timeout','look_ahead_dist','curve_beta'):
#             if attr in new: setattr(self, attr, new[attr])
#         if 'num_curve_samples' in new: self.num_curve_samp = int(new['num_curve_samples'])
#         if 'A' in new: self.traj.A = new['A']
#         return SetParametersResult(successful=True)

#     def _startup_banner(self, p):
#         return f"Khởi động HA Figure-8 (MIT Tuned) | v_path_max={p['v_path_max']} | Ôm cua mượt!"

# # ===========================================================================
# def main():
#     rclpy.init()
#     node = Figure8HANode()
#     try: rclpy.spin(node)
#     except KeyboardInterrupt: pass
#     finally:
#         node._stop()
#         node._save_csv()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
