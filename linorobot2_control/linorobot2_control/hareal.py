#!/usr/bin/env python3
"""
hareal.py  –  HA Figure-8 Trajectory Tracking  |  REAL ROBOT  |  ROS2
=======================================================================
Cấu trúc ROS2 căn theo code Lyapunov thực tế (gọn, sạch, không thừa).
Thuật toán HA giữ nguyên từ bản mô phỏng đã kiểm chứng (ha_figure8_sim.py).

Thay đổi so với hareal.py cũ:
  - Bỏ: watchdog timer, odom ready-count guard, pose LPF, parameter
    descriptor FloatingPointRange, SetParametersResult callback
  - Giữ: use_sim_time=False, RELIABLE QoS cho /cmd_vel
  - Cấu trúc class/method căn theo RealRobotLyapunovFollower
  - Thuật toán HA (HedgeAlgebra, HAController, rule bases, Figure8Trajectory)
    lấy nguyên từ sim - đúng công thức, đúng giá trị MAX_E/gain/lpf
"""

import csv
import datetime
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
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
            if name not in {'Zero', 'Neutral', 'Absolute', 'Small', 'Large'} else 0
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
        'Zero':     'Zero',    'VVSmall':  'Zero',     'VSmall':   'VVSmall',
        'LVSmall':  'VSmall',  'Small':    'VSmall',   'VLSmall':  'Small',
        'LSmall':   'LSmall',  'LLSmall':  'Large',    'Neutral':  'Large',
        'LLLarge':  'VLarge',  'LLarge':   'VLarge',   'VLLarge':  'VVLarge',
        'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
        'VVLarge':  'Absolute','Absolute': 'Absolute',
    }

def _rule_lateral():
    return {
        'Zero':     'Zero',    'VVSmall':  'Zero',     'VSmall':   'Zero',
        'LVSmall':  'VVSmall', 'Small':    'VSmall',   'VLSmall':  'VSmall',
        'LSmall':   'Small',   'LLSmall':  'LSmall',   'Neutral':  'Large',
        'LLLarge':  'Large',   'LLarge':   'VLarge',   'VLLarge':  'VLarge',
        'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
        'VVLarge':  'Absolute','Absolute': 'Absolute',
    }

def _rule_heading():
    return {
        'Zero':     'Zero',    'VVSmall':  'Zero',     'VSmall':   'VVSmall',
        'LVSmall':  'VSmall',  'Small':    'Small',    'VLSmall':  'Small',
        'LSmall':   'LSmall',  'LLSmall':  'Large',    'Neutral':  'Large',
        'LLLarge':  'VLarge',  'LLarge':   'VLarge',   'VLLarge':  'VVLarge',
        'Large':    'VVLarge', 'LVLarge':  'Absolute', 'VLarge':   'Absolute',
        'VVLarge':  'Absolute','Absolute': 'Absolute',
    }


# ===========================================================================
#  HA CONTROLLER  -  tham so lay nguyen tu sim da kiem chung
# ===========================================================================
class HAController:
    # Dai chuan hoa thu hep: khai thac day du luat HA o sai so nho
    MAX_E1 = 0.040          # sai so doc
    MAX_E2 = 0.040          # sai so ngang  <- chu chot giam lech ngang
    MAX_E3 = math.pi / 4    # sai so heading

    def __init__(self, ha,
                 gain_e1=0.25, gain_e2=1.40, gain_e3=1.80,
                 v_max=0.33, w_max=3.85, dv_max=2.00, dw_max=10.00,
                 Ts=0.05, lpf_alpha=0.25):
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

        # LPF goc: giam tre pha → robot re kip hon
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
#  FIGURE-8 TRAJECTORY
# ===========================================================================
class Figure8Trajectory:

    def __init__(self, A, off_x, off_y, L_loop=30.0):
        self.A      = A
        self.off_x  = off_x
        self.off_y  = off_y
        self.L_loop = L_loop
        self.freq   = 2.0 * math.pi / L_loop

    def pos(self, s):
        f = self.freq
        return (self.off_x + self.A * math.sin(f * s),
                self.off_y + self.A * math.sin(2 * f * s))

    def curvature(self, s):
        f, A = self.freq, self.A
        dx   =  A * f      * math.cos(f * s)
        dy   =  2 * A * f  * math.cos(2 * f * s)
        ddx  = -A * f**2   * math.sin(f * s)
        ddy  = -4 * A * f**2 * math.sin(2 * f * s)
        return abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-9)

    def ref_kinematics(self, s, v_s, a_s):
        f, A   = self.freq, self.A
        dx_ds  =  A * f      * math.cos(f * s)
        dy_ds  =  2 * A * f  * math.cos(2 * f * s)
        ddx_ds = -A * f**2   * math.sin(f * s)
        ddy_ds = -4 * A * f**2 * math.sin(2 * f * s)
        dx_dt  = dx_ds * v_s
        dy_dt  = dy_ds * v_s
        ddx_dt = ddx_ds * v_s**2 + dx_ds * a_s
        ddy_dt = ddy_ds * v_s**2 + dy_ds * a_s
        phi_ref = math.atan2(dy_dt, dx_dt)
        v_ref   = math.hypot(dx_dt, dy_dt)
        denom   = dx_dt**2 + dy_dt**2 + 1e-9
        w_ref   = (dx_dt * ddy_dt - dy_dt * ddx_dt) / denom
        return phi_ref, v_ref, w_ref

    def sample_path(self, n_pts=600):
        return [self.pos(i / n_pts * self.L_loop) for i in range(n_pts + 1)]


# ===========================================================================
#  MAIN ROS2 NODE  -  REAL ROBOT
#  Cau truc can theo RealRobotLyapunovFollower
# ===========================================================================
class Figure8HARealNode(Node):

    def __init__(self):
        super().__init__(
            'ha_figure8_real',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, False)
            ]
        )

        # ------------------------------------------------------------------
        # PARAMETERS  -  cung convention voi Lyapunov real
        # ------------------------------------------------------------------
        self.declare_parameter('offset_x',   1.1)
        self.declare_parameter('offset_y',   0.8)
        self.declare_parameter('offset_phi', 0.0)
        self.declare_parameter('Ts',         0.05)
        self.declare_parameter('num_loops',  3)
        self.declare_parameter('A',          0.7)
        self.declare_parameter('v_path_max', 0.4)
        self.declare_parameter('a_path_max', 0.2)

        self.off_x      = self.get_parameter('offset_x').value
        self.off_y      = self.get_parameter('offset_y').value
        self.off_phi    = self.get_parameter('offset_phi').value
        self.Ts         = self.get_parameter('Ts').value
        self.num_loops  = self.get_parameter('num_loops').value
        self.A          = self.get_parameter('A').value
        self.v_path_max = self.get_parameter('v_path_max').value
        self.a_path_max = self.get_parameter('a_path_max').value

        self.get_logger().info(
            f'REAL ROBOT HA | Ts={self.Ts}s | '
            f'offset=({self.off_x}, {self.off_y}, {self.off_phi:.2f}rad) | '
            f'A={self.A}m  loops={self.num_loops}  v_path={self.v_path_max}m/s'
        )

        # ------------------------------------------------------------------
        # TRAJECTORY  &  CONTROLLER
        # ------------------------------------------------------------------
        self.s_end_one_loop = 30.0
        self.s_end  = self.s_end_one_loop * self.num_loops
        self.traj   = Figure8Trajectory(
            A=self.A, off_x=self.off_x, off_y=self.off_y,
            L_loop=self.s_end_one_loop)

        self.ha   = HedgeAlgebra()
        self.ctrl = HAController(
            ha=self.ha,
            gain_e1=0.25, gain_e2=1.40, gain_e3=1.80,
            v_max=0.33,   w_max=3.85,
            dv_max=2.00,  dw_max=10.00,
            Ts=self.Ts,   lpf_alpha=0.25)

        self.get_logger().info(self.ha.sem_table_str())

        # ------------------------------------------------------------------
        # LOOK-AHEAD (adaptive speed, giu tu sim)
        # ------------------------------------------------------------------
        self.look_ahead_dist = 0.50
        self.curve_beta      = 0.06
        self.num_curve_samp  = 8

        # ------------------------------------------------------------------
        # STATE VARIABLES  -  giong Lyapunov
        # ------------------------------------------------------------------
        self.s             = 0.0
        self.v_s           = 0.02
        self.curr_q        = [self.off_x, self.off_y, self.off_phi]
        self.odom_received = False
        self.start_time    = None
        self.odom_timeout  = 0.5
        self.last_odom_t   = None

        # Logging
        self.sum_e_dist_sq = 0.0
        self.count         = 0
        self.last_log_t    = -1
        self.data_log      = []
        self.csv_filename  = (
            f'hareal_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )

        # ------------------------------------------------------------------
        # ROS INFRASTRUCTURE  -  QoS can theo Lyapunov, fix RELIABLE cmd_vel
        # ------------------------------------------------------------------
        # cmd_vel RELIABLE: tranh QoS mismatch voi robot driver
        qos_cmdvel = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        # ref_path latched
        qos_latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.cmd_pub      = self.create_publisher(Twist, '/cmd_vel',    qos_cmdvel)
        self.act_path_pub = self.create_publisher(Path,  '/robot_path', 10)
        self.ref_path_pub = self.create_publisher(Path,  '/ref_path',   qos_latch)
        self.odom_sub     = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.actual_path = Path()
        self.actual_path.header.frame_id = 'world'

        # ------------------------------------------------------------------
        # TIMERS  -  giong Lyapunov
        # ------------------------------------------------------------------
        self.ref_timer = self.create_timer(5.0,     self.publish_static_ref_path)
        self.timer     = self.create_timer(self.Ts, self.control_loop)

        self.get_logger().info('HA Figure-8 Real Robot Node Ready.')

    # =======================================================================
    #  REF PATH  (publish moi 5s, latched)
    # =======================================================================
    def publish_static_ref_path(self):
        ref_path = Path()
        ref_path.header.frame_id = 'world'
        ref_path.header.stamp    = self.get_clock().now().to_msg()
        for i in range(600):
            ss = (i / 600.0) * self.s_end_one_loop
            px, py = self.traj.pos(ss)
            ps = PoseStamped()
            ps.header.frame_id = 'world'
            ps.pose.position.x = px
            ps.pose.position.y = py
            ref_path.poses.append(ps)
        self.ref_path_pub.publish(ref_path)

    # =======================================================================
    #  ODOM CALLBACK  -  gon nhu Lyapunov, offset transform giu tu sim
    # =======================================================================
    def odom_callback(self, msg: Odometry):
        self.last_odom_t = self.get_clock().now()

        ox  = msg.pose.pose.position.x
        oy  = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        ophi = math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y**2 + q.z**2))

        c, s_ = math.cos(self.off_phi), math.sin(self.off_phi)
        self.curr_q[0] = ox * c - oy * s_ + self.off_x
        self.curr_q[1] = ox * s_ + oy * c  + self.off_y
        self.curr_q[2] = math.atan2(math.sin(ophi + self.off_phi),
                                    math.cos(ophi + self.off_phi))
        self.odom_received = True

    # =======================================================================
    #  ADAPTIVE SPEED  -  giu tu sim
    # =======================================================================
    def _v_safe(self, s):
        v_min = self.v_path_max
        n = max(self.num_curve_samp, 1)
        for i in range(n + 1):
            kap  = self.traj.curvature(s + i / n * self.look_ahead_dist)
            v_ok = self.v_path_max / (1.0 + self.curve_beta * kap)
            if v_ok < v_min:
                v_min = v_ok
        return max(v_min, 0.03)

    # =======================================================================
    #  CONTROL LOOP  -  cau truc giong Lyapunov, noi dung HA tu sim
    # =======================================================================
    def control_loop(self):
        if not self.odom_received:
            return

        now = self.get_clock().now()

        # Kiem tra odom timeout
        if self.last_odom_t is not None:
            dt_odom = (now - self.last_odom_t).nanoseconds * 1e-9
            if dt_odom > self.odom_timeout:
                self.get_logger().warn(
                    f'Mat /odom {dt_odom:.2f}s! Dung khan cap.',
                    throttle_duration_sec=2.0)
                self.stop_robot()
                return

        if self.start_time is None:
            self.start_time = now
            self.get_logger().info('=== BAT DAU BAM QUY DAO ===')
            return

        t = (now - self.start_time).nanoseconds * 1e-9

        # A. MOTION LAW s(t)  -  giu tu sim (dist_brake profile)
        v_adapt    = self._v_safe(self.s)
        dist_brake = self.v_s**2 / (2 * self.a_path_max + 1e-9)
        dist_left  = self.s_end - self.s

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

        if self.s >= self.s_end:
            self.stop_robot()
            self.save_csv()
            self.get_logger().info('=== HOAN THANH QUY DAO ===')
            return

        # B. FEEDFORWARD  -  ref kinematics tu sim
        phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
        x_ref, y_ref = self.traj.pos(self.s)

        # C. SAI SO trong frame body robot
        rx, ry, rphi = self.curr_q
        dx_g = x_ref - rx
        dy_g = y_ref - ry
        cq = math.cos(rphi)
        sq = math.sin(rphi)
        e1 =  cq * dx_g + sq * dy_g
        e2 = -sq * dx_g + cq * dy_g
        e3 = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))

        # D. HA FEEDBACK
        v_cmd, w_cmd, dbg = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

        # E. PUBLISH
        cmd = Twist()
        cmd.linear.x  = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

        # F. LOG & VIZ
        e_dist = math.hypot(dx_g, dy_g)
        self.update_path_and_log(t, e_dist, e1, e2, e3,
                                 v_cmd, w_cmd, v_ref, w_ref, dbg)

    # =======================================================================
    #  STOP  -  giong Lyapunov
    # =======================================================================
    def stop_robot(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())
        self.get_logger().info('=== ROBOT DUNG ===', once=True)

    # =======================================================================
    #  LOG & ACTUAL PATH  -  giong Lyapunov, them cot HA
    # =======================================================================
    def update_path_and_log(self, t, e_dist, e1, e2, e3,
                            v, w, v_ref, w_ref, dbg):
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x = self.curr_q[0]
        pose.pose.position.y = self.curr_q[1]
        self.actual_path.poses.append(pose)
        if len(self.actual_path.poses) > 3000:
            self.actual_path.poses.pop(0)
        self.act_path_pub.publish(self.actual_path)

        self.sum_e_dist_sq += e_dist**2
        self.count         += 1
        rms_err = math.sqrt(self.sum_e_dist_sq / self.count)

        self.data_log.append([
            round(t, 3),
            round(e_dist, 5), round(rms_err, 5),
            round(e1, 5),     round(e2, 5),    round(e3, 5),
            round(v, 5),      round(w, 5),
            round(v_ref, 5),  round(w_ref, 5),
            round(dbg['fb1'], 5), round(dbg['fb2'], 5), round(dbg['fb3'], 5),
            round(self.curr_q[0], 4), round(self.curr_q[1], 4),
        ])

        if int(t) > self.last_log_t:
            self.get_logger().info(
                f't={t:.1f}s | es={e_dist:.4f}m RMS={rms_err:.4f}m | '
                f'e2={e2:.4f}m e3={math.degrees(e3):.2f}deg | '
                f'v={v:.4f} w={w:.4f}')
            self.last_log_t = int(t)

    # =======================================================================
    #  SAVE CSV  -  giong Lyapunov
    # =======================================================================
    def save_csv(self):
        if not self.data_log:
            return
        with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time(s)',    'es(m)',       'RMSE(m)',
                'e1_long(m)', 'e2_lat(m)',   'e3_head(rad)',
                'v_cmd(m/s)', 'w_cmd(rad/s)',
                'v_ref(m/s)', 'w_ref(rad/s)',
                'fb1',        'fb2',         'fb3',
                'robot_x(m)', 'robot_y(m)',
            ])
            writer.writerows(self.data_log)
        self.get_logger().info(f'Da luu log: {self.csv_filename}')


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    rclpy.init()
    node = Figure8HARealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Nhan Ctrl+C → dung robot.')
    finally:
        node.stop_robot()
        node.save_csv()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()