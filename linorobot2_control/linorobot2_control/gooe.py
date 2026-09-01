#!/usr/bin/env python3
"""
ha_figure8_goa_gazebo.py  –  HA Figure-8 + GOOSE OPTIMIZATION ALGORITHM (GOA) | GAZEBO ROS2
=============================================================================================
Tích hợp thuật toán Bầy Ngỗng (GOA) CHUẨN XÁC để Auto-Tune 10 thông số điều khiển HA
trực tiếp trong môi trường Gazebo. Tự động reset world sau mỗi vòng lặp.

GOA đúng chuẩn gồm 2 pha:
  Phase 1 – EXPLORATION  : Lévy-flight foraging (ngỗng tìm kiếm thức ăn ngẫu nhiên)
  Phase 2 – EXPLOITATION : V-formation flight   (ngỗng bay theo đội hình V bám leader)

Tài liệu tham khảo:
  • "Goose Optimization Algorithm: A novel nature-inspired optimizer for solving
     engineering problems" – (GOA, 2023-2024)
  • Đặc trưng: góc V θ thu hẹp dần, hệ số draft cos(θ), Lévy flight exponent β=1.5
"""

import csv
import datetime
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Empty

# ===========================================================================
#  HEDGE ALGEBRA (Giữ nguyên toán học gốc)
# ===========================================================================
class HedgeAlgebra:
    def __init__(self, theta=0.5, alpha=0.7, beta=0.3):
        self.theta, self.alpha, self.beta = theta, alpha, beta
        self.fm = {'V': alpha, 'L': beta, 'Small': theta, 'Large': 1.0 - theta}
        self.term_specs = [
            ('Zero', [], None), ('VVSmall', ['V','V'], 'Small'), ('VSmall', ['V'], 'Small'),
            ('LVSmall', ['L','V'], 'Small'), ('Small', [], 'Small'), ('VLSmall', ['V','L'], 'Small'),
            ('LSmall', ['L'], 'Small'), ('LLSmall', ['L','L'], 'Small'), ('Neutral', [], None),
            ('LLLarge', ['L','L'], 'Large'), ('LLarge', ['L'], 'Large'), ('VLLarge', ['V','L'], 'Large'),
            ('Large', [], 'Large'), ('LVLarge', ['L','V'], 'Large'), ('VLarge', ['V'], 'Large'),
            ('VVLarge', ['V','V'], 'Large'), ('Absolute', [], None)
        ]
        self.sem = {n: self._compute_sem(n, h, b) for n, h, b in self.term_specs}
        self.ordered_terms = sorted(self.sem, key=lambda t: self.sem[t])
        self.hcnt = {n: len(h) if n not in {'Zero','Neutral','Absolute','Small','Large'} else 0 for n, h, _ in self.term_specs}

    def _sgn_hedge(self, h): return 1 if h == 'V' else -1
    def _sgn_base(self, h, base): return (-1 if h == 'V' else 1) if base == 'Small' else (1 if h == 'V' else -1)

    def _compute_sem(self, name, hedges, base):
        if name == 'Zero': return 0.0
        if name == 'Neutral': return self.theta
        if name == 'Absolute': return 1.0
        fm_base = self.fm[base]
        current_v = self.theta - self.alpha * fm_base if base == 'Small' else self.theta + self.alpha * fm_base
        if not hedges: return current_v
        current_fm = fm_base
        for h in reversed(hedges):
            fm_hx = (self.beta if h == 'V' else self.alpha) * current_fm
            sum_fm = fm_hx if h == 'V' else current_fm
            current_v += self._sgn_base(h, base) * (sum_fm - 0.5 * (1 + self._sgn_hedge(h) * (self.beta - self.alpha)) * fm_hx)
            current_fm = fm_hx
        return current_v

    def fuzzify(self, value):
        v = float(np.clip(value, 0.0, 1.0))
        min_d = min(abs(self.sem[t] - v) for t in self.ordered_terms)
        cands = [t for t in self.ordered_terms if abs(self.sem[t] - v) <= min_d + 1e-9]
        return min(cands, key=lambda t: self.hcnt.get(t, 3))

    def infer(self, input_val, rule_base):
        return self.sem[rule_base.get(self.fuzzify(float(np.clip(input_val, 0.0, 1.0))), 'Neutral')]

def _rule_longitudinal(): return {'Zero':'Zero', 'VVSmall':'Zero', 'VSmall':'VVSmall', 'LVSmall':'VSmall', 'Small':'VSmall', 'VLSmall':'Small', 'LSmall':'LSmall', 'LLSmall':'Large', 'Neutral':'Large', 'LLLarge':'VLarge', 'LLarge':'VLarge', 'VLLarge':'VVLarge', 'Large':'VVLarge', 'LVLarge':'Absolute', 'VLarge':'Absolute', 'VVLarge':'Absolute', 'Absolute':'Absolute'}
def _rule_lateral(): return {'Zero':'Zero', 'VVSmall':'Zero', 'VSmall':'Zero', 'LVSmall':'VVSmall', 'Small':'VSmall', 'VLSmall':'VSmall', 'LSmall':'Small', 'LLSmall':'LSmall', 'Neutral':'Large', 'LLLarge':'Large', 'LLarge':'VLarge', 'VLLarge':'VLarge', 'Large':'VVLarge', 'LVLarge':'Absolute', 'VLarge':'Absolute', 'VVLarge':'Absolute', 'Absolute':'Absolute'}
def _rule_heading(): return {'Zero':'Zero', 'VVSmall':'Zero', 'VSmall':'VVSmall', 'LVSmall':'VSmall', 'Small':'Small', 'VLSmall':'Small', 'LSmall':'LSmall', 'LLSmall':'Large', 'Neutral':'Large', 'LLLarge':'VLarge', 'LLarge':'VLarge', 'VLLarge':'VVLarge', 'Large':'VVLarge', 'LVLarge':'Absolute', 'VLarge':'Absolute', 'VVLarge':'Absolute', 'Absolute':'Absolute'}

# ===========================================================================
#  HA CONTROLLER
# ===========================================================================
class HAController:
    def __init__(self, Ts=0.05, v_max=0.55, w_max=4.50, dv_max=2.00, dw_max=10.00):
        self.v_max, self.w_max, self.dv_max, self.dw_max, self.Ts = v_max, w_max, dv_max, dw_max, Ts
        self.rule_long, self.rule_lat, self.rule_head = _rule_longitudinal(), _rule_lateral(), _rule_heading()
        self.reset()

    def set_params(self, p):
        # p = [gain_e1, gain_e2, gain_e3, MAX_E1, MAX_E2, MAX_E3, alpha, theta, lpf_alpha, look_ahead]
        self.gain_e1, self.gain_e2, self.gain_e3 = p[0], p[1], p[2]
        self.MAX_E1, self.MAX_E2, self.MAX_E3    = p[3], p[4], p[5]
        self.alpha, self.theta  = p[6], p[7]
        self.lpf_alpha          = p[8]
        self.look_ahead_dist    = p[9]
        self.ha = HedgeAlgebra(theta=self.theta, alpha=self.alpha, beta=1.0 - self.alpha)

    def reset(self):
        self.v_prev = self.w_prev = self.w_lpf_prev = 0.0

    def _ha_fb(self, err, max_e, rule, gain):
        if abs(err) <= 0.01 * max_e: return 0.0
        return math.copysign(self.ha.infer(min(abs(err) / max_e, 1.0), rule) * gain, err)

    def compute(self, v_ref, w_ref, e1, e2, e3):
        fb1 = self._ha_fb(e1, self.MAX_E1, self.rule_long, self.gain_e1)
        fb2 = self._ha_fb(e2, self.MAX_E2, self.rule_lat,  self.gain_e2)
        fb3 = self._ha_fb(e3, self.MAX_E3, self.rule_head, self.gain_e3)
        v_raw  = v_ref * math.cos(e3) + fb1
        w_raw  = w_ref + fb2 + fb3
        w_filt = self.lpf_alpha * self.w_lpf_prev + (1 - self.lpf_alpha) * w_raw
        self.w_lpf_prev = w_filt
        v_sat = float(np.clip(v_raw,  -self.v_max, self.v_max))
        w_sat = float(np.clip(w_filt, -self.w_max, self.w_max))
        dv = float(np.clip(v_sat - self.v_prev, -self.dv_max * self.Ts, self.dv_max * self.Ts))
        dw = float(np.clip(w_sat - self.w_prev, -self.dw_max * self.Ts, self.dw_max * self.Ts))
        v_cmd, w_cmd = self.v_prev + dv, self.w_prev + dw
        self.v_prev, self.w_prev = v_cmd, w_cmd
        return v_cmd, w_cmd

# ===========================================================================
#  FIGURE-8 TRAJECTORY
# ===========================================================================
class Figure8Trajectory:
    def __init__(self, A, off_x, off_y, L_loop=30.0):
        self.A, self.off_x, self.off_y, self.L_loop = A, off_x, off_y, L_loop
        self.freq = 2.0 * math.pi / L_loop

    def pos(self, s):
        return (self.off_x + self.A * math.sin(self.freq * s),
                self.off_y + self.A * math.sin(2 * self.freq * s))

    def curvature(self, s):
        dx  = self.A * self.freq   * math.cos(self.freq * s)
        dy  = 2 * self.A * self.freq   * math.cos(2 * self.freq * s)
        ddx = -self.A * self.freq**2   * math.sin(self.freq * s)
        ddy = -4 * self.A * self.freq**2 * math.sin(2 * self.freq * s)
        return abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-9)

    def ref_kinematics(self, s, v_s, a_s):
        dx_ds  = self.A * self.freq   * math.cos(self.freq * s)
        dy_ds  = 2 * self.A * self.freq   * math.cos(2 * self.freq * s)
        ddx_ds = -self.A * self.freq**2   * math.sin(self.freq * s)
        ddy_ds = -4 * self.A * self.freq**2 * math.sin(2 * self.freq * s)
        dx_dt  = dx_ds * v_s
        dy_dt  = dy_ds * v_s
        ddx_dt = ddx_ds * v_s**2 + dx_ds * a_s
        ddy_dt = ddy_ds * v_s**2 + dy_ds * a_s
        phi    = math.atan2(dy_dt, dx_dt)
        v      = math.hypot(dx_dt, dy_dt)
        w      = (dx_dt * ddy_dt - dy_dt * ddx_dt) / (dx_dt**2 + dy_dt**2 + 1e-9)
        return phi, v, w


# ===========================================================================
#  GOOSE OPTIMIZATION ALGORITHM – CHUẨN XÁC
# ===========================================================================
class GOA:
    """
    Goose Optimization Algorithm (GOA) – Chuẩn xác theo nguyên lý sinh học:

    Hai pha chính:
    ─────────────────────────────────────────────────────────────
    1. EXPLORATION – Foraging (pha tìm kiếm):
       Ngỗng bay ngẫu nhiên tìm thức ăn, dùng Lévy flight để
       mô phỏng bước nhảy dài-ngắn xen kẽ (heavy-tail distribution).
       Điều kiện kích hoạt: |A| >= 1
       Công thức:
         X_rand = vị trí ngỗng ngẫu nhiên trong đàn
         D      = |C · X_rand − X_i|
         X_new  = X_rand − A · D + Lévy · σ

    2. EXPLOITATION – V-formation flight (pha khai thác):
       Ngỗng bay trong đội hình V bám theo leader.
       Góc đội hình V: θ_v thu hẹp dần theo iteration → hội tụ.
       Hệ số draft cos(θ_v) = lợi thế khí động học.
       Điều kiện kích hoạt: |A| < 1
       Công thức:
         D      = |C · X_leader − X_i|
         X_new  = X_leader − A · D · cos(θ_v)

    Tham số chính:
    ─────────────────────────────────────────────────────────────
    a   : giảm tuyến tính từ 2 → 0 theo thế hệ
          (điều tiết biên độ bước nhảy)
    A   = 2·a·r1 − a  (vector ngẫu nhiên, phụ thuộc a)
    C   = 2·r2        (hệ số tương tác xã hội)
    θ_v = π/6 · (1 − t/T)  (góc V hẹp dần: 30° → 0°)
    cos(θ_v)            (draft aerodynamic factor)
    Lévy(β=1.5)         (heavy-tail random walk)
    σ   = step scale giảm dần theo t/T
    """

    LEVY_BETA = 1.5   # Chỉ số Lévy (1 < β ≤ 2 cho heavy-tail)
    V_ANGLE_MAX = math.pi / 6   # Góc V tối đa = 30°

    def __init__(self, num_geese: int, num_gens: int, dim: int, bounds: np.ndarray):
        self.num_geese = num_geese
        self.num_gens  = num_gens
        self.dim       = dim
        self.bounds    = bounds   # shape (dim, 2)

        # Hệ số Lévy – tính một lần, dùng lại
        b = self.LEVY_BETA
        self._levy_sigma = (
            math.gamma(1 + b) * math.sin(math.pi * b / 2) /
            (math.gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))
        ) ** (1 / b)

        # Khởi tạo quần thể ngẫu nhiên
        lo, hi = bounds[:, 0], bounds[:, 1]
        self.population = np.random.uniform(lo, hi, (num_geese, dim))
        self.fitness    = np.full(num_geese, np.inf)

        # Leader toàn cục
        self.leader_pos   = None
        self.leader_score = np.inf

        self.generation = 0       # thế hệ hiện tại (0-indexed)

    # ------------------------------------------------------------------
    #  Lévy flight step
    # ------------------------------------------------------------------
    def _levy_step(self) -> np.ndarray:
        """
        Sinh bước Lévy theo thuật toán Mantegna (1994):
          step = u / |v|^(1/β),  u~N(0,σ²),  v~N(0,1)
        Lévy flight tạo bước đi dài ngẫu nhiên xen kẽ bước ngắn,
        giúp thoát khỏi cực tiểu địa phương.
        """
        u = np.random.randn(self.dim) * self._levy_sigma
        v = np.abs(np.random.randn(self.dim)) + 1e-9
        return u / (v ** (1.0 / self.LEVY_BETA))

    # ------------------------------------------------------------------
    #  Cập nhật leader
    # ------------------------------------------------------------------
    def update_leader(self):
        """Cập nhật leader toàn cục từ quần thể hiện tại."""
        best_idx = int(np.argmin(self.fitness))
        if self.fitness[best_idx] < self.leader_score:
            self.leader_score = float(self.fitness[best_idx])
            self.leader_pos   = self.population[best_idx].copy()

    # ------------------------------------------------------------------
    #  Bước tiến hóa một thế hệ – ĐÂY LÀ PHẦN CHUẨN XÁC NHẤT CỦA GOA
    # ------------------------------------------------------------------
    def evolve(self):
        """
        Thực thi một thế hệ GOA. Gọi sau khi đã đánh giá fitness toàn bộ đàn.

        Quy trình:
          1. Cập nhật leader (best solution so far)
          2. Tính hệ số thích nghi a, θ_v cho thế hệ này
          3. Với từng cá thể ngỗng:
             - Nếu |A| >= 1 → Exploration (Lévy foraging)
             - Nếu |A| <  1 → Exploitation (V-formation)
          4. Clip vào bounds
          5. Tăng bộ đếm thế hệ
        """
        if self.leader_pos is None:
            self.update_leader()

        t = self.generation          # thế hệ hiện tại
        T = self.num_gens            # tổng số thế hệ

        # ── Hệ số tuyến tính: a giảm từ 2 → 0 ──────────────────────────
        # a điều tiết biên độ bước nhảy của từng ngỗng
        a = 2.0 * (1.0 - t / T)

        # ── Góc đội hình V: θ_v thu hẹp từ π/6 (30°) → 0 ──────────────
        # Khi đàn hội tụ, ngỗng bay sát nhau hơn → khai thác tốt hơn
        theta_v = self.V_ANGLE_MAX * (1.0 - t / T)
        draft   = math.cos(theta_v)   # hệ số lợi thế khí động (0 < draft ≤ 1)

        # ── Hệ số bước Lévy giảm dần (exploration yếu đi theo thế hệ) ──
        levy_scale = 0.5 * (1.0 - t / T)

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        new_pop = np.empty_like(self.population)

        for i in range(self.num_geese):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            # Vector A – quyết định exploration vs exploitation
            A = 2.0 * a * r1 - a          # phần tử thuộc (-2a, 2a)
            C = 2.0 * r2                   # [0, 2]

            A_norm = float(np.linalg.norm(A))

            if A_norm >= 1.0:
                # ══════════════════════════════════════════════════
                # PHA EXPLORATION: Lévy-flight foraging
                # Ngỗng bay xa để tìm vùng thức ăn mới
                # Chọn ngỗng ngẫu nhiên (không nhất thiết là leader)
                # ══════════════════════════════════════════════════
                rand_idx = np.random.randint(0, self.num_geese)
                X_rand   = self.population[rand_idx]

                D_rand   = np.abs(C * X_rand - self.population[i])
                step_dir = X_rand - A * D_rand

                # Cộng thêm Lévy flight để tránh bẫy cực tiểu địa phương
                levy     = self._levy_step()
                new_pos  = step_dir + levy_scale * levy

            else:
                # ══════════════════════════════════════════════════
                # PHA EXPLOITATION: V-formation flight
                # Ngỗng bay sát leader trong đội hình V
                # draft = cos(θ_v) mô phỏng lợi thế khí động học
                # ══════════════════════════════════════════════════
                D_leader = np.abs(C * self.leader_pos - self.population[i])
                new_pos  = self.leader_pos - A * D_leader * draft

            new_pop[i] = np.clip(new_pos, lo, hi)

        self.population = new_pop
        self.fitness[:] = np.inf    # Reset fitness, chờ đánh giá lại
        self.generation += 1

    # ------------------------------------------------------------------
    #  Tiện ích
    # ------------------------------------------------------------------
    def is_done(self) -> bool:
        return self.generation >= self.num_gens

    def set_fitness(self, idx: int, value: float):
        self.fitness[idx] = value

    def get_individual(self, idx: int) -> np.ndarray:
        return self.population[idx]

    def summary(self) -> str:
        return (f"Gen {self.generation}/{self.num_gens} | "
                f"Best RMSE = {self.leader_score * 1000:.2f} mm | "
                f"Leader = [{', '.join(f'{v:.3f}' for v in self.leader_pos)}]")


# ===========================================================================
#  MAIN ROS2 NODE
# ===========================================================================
class Figure8GooseTunerNode(Node):
    L_LOOP = 30.0

    def __init__(self):
        super().__init__(
            'ha_figure8_goose_tuner',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        # ── Thông số vật lý ────────────────────────────────────────────
        self.Ts           = 0.05
        self.v_path_max   = 0.60
        self.a_path_max   = 0.35
        self.curve_beta   = 0.06
        self.s_end        = self.L_LOOP * 1.5   # 1.5 vòng đủ tính RMSE

        self.traj = Figure8Trajectory(A=0.7, off_x=0.0, off_y=0.0, L_loop=self.L_LOOP)
        self.ctrl = HAController(Ts=self.Ts, v_max=0.65, w_max=4.50)

        # ── Cấu hình GOA ───────────────────────────────────────────────
        NUM_GEESE = 10
        NUM_GENS  = 10
        DIM       = 10
        BOUNDS    = np.array([
            [0.10, 0.80],   # gain_e1
            [0.80, 3.00],   # gain_e2
            [1.00, 3.00],   # gain_e3
            [0.02, 0.08],   # MAX_E1
            [0.02, 0.08],   # MAX_E2
            [0.20, 0.80],   # MAX_E3
            [0.60, 0.85],   # HA alpha
            [0.40, 0.60],   # HA theta
            [0.10, 0.50],   # lpf_alpha
            [0.30, 0.70],   # look_ahead
        ])

        # Khởi tạo đối tượng GOA – toàn bộ logic tối ưu nằm trong class GOA
        self.goa = GOA(
            num_geese=NUM_GEESE,
            num_gens=NUM_GENS,
            dim=DIM,
            bounds=BOUNDS
        )

        # ── Trạng thái FSM ─────────────────────────────────────────────
        #  INIT_GOOSE → RUNNING → (reset) → INIT_GOOSE → ... → FINISHED
        self.state            = 'INIT_GOOSE'
        self.curr_goose_idx   = 0
        self.s                = 0.0
        self.v_s              = 0.02
        self.curr_q           = [0.0, 0.0, 0.0]
        self.sum_es_sq        = 0.0
        self.count            = 0
        self.reset_wait_cycles = 0

        # ── ROS2 I/O ───────────────────────────────────────────────────
        self.cmd_pub      = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub     = self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.reset_client = self.create_client(Empty, '/reset_simulation')

        self.create_timer(self.Ts, self._fsm_loop)

        self.get_logger().info('🚀 KHỞI ĐỘNG AUTO-TUNING GAZEBO BẰNG GOA CHUẨN XÁC!')
        self.get_logger().info(
            f'   Cấu hình: {NUM_GEESE} ngỗng × {NUM_GENS} thế hệ = '
            f'{NUM_GEESE * NUM_GENS} lần chạy Gazebo'
        )
        self.get_logger().info(
            '   Gợi ý: set <real_time_update_rate>0</...> trong world '
            'Gazebo để chạy siêu tốc.'
        )

    # ------------------------------------------------------------------
    #  Odometry callback
    # ------------------------------------------------------------------
    def _odom_cb(self, msg):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        q  = msg.pose.pose.orientation
        phi = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y**2 + q.z**2)
        )
        self.curr_q = [ox, oy, phi]

    # ------------------------------------------------------------------
    #  Reset Gazebo world
    # ------------------------------------------------------------------
    def _reset_gazebo_world(self):
        self.cmd_pub.publish(Twist())
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Đang đợi /reset_simulation service...')
        self.reset_client.call_async(Empty.Request())
        # Chờ 0.5s để robot ổn định sau khi teleport
        self.reset_wait_cycles = int(0.5 / self.Ts)
        self.state = 'RESETTING'

    # ------------------------------------------------------------------
    #  Tính vận tốc an toàn theo độ cong quỹ đạo
    # ------------------------------------------------------------------
    def _v_safe(self, s):
        kap = self.traj.curvature(s + self.ctrl.look_ahead_dist)
        return max(self.v_path_max / (1.0 + self.curve_beta * kap), 0.03)

    # ------------------------------------------------------------------
    #  FSM loop chính
    # ------------------------------------------------------------------
    def _fsm_loop(self):

        # ── FINISHED ──────────────────────────────────────────────────
        if self.state == 'FINISHED':
            self.cmd_pub.publish(Twist())
            self.get_logger().info('✅ TỐI ƯU THÀNH CÔNG! BỘ THÔNG SỐ VÔ ĐỊCH:')
            names = [
                'gain_e1', 'gain_e2', 'gain_e3',
                'MAX_E1',  'MAX_E2',  'MAX_E3',
                'alpha',   'theta',   'lpf_alpha', 'look_ahead'
            ]
            for n, v in zip(names, self.goa.leader_pos):
                self.get_logger().info(f'   {n:<12} = {v:.4f}')
            self.get_logger().info(
                f'   RMSE tốt nhất = {self.goa.leader_score * 1000:.2f} mm'
            )
            rclpy.shutdown()
            return

        # ── RESETTING ─────────────────────────────────────────────────
        if self.state == 'RESETTING':
            self.reset_wait_cycles -= 1
            if self.reset_wait_cycles <= 0:
                self.state = 'INIT_GOOSE'
            return

        # ── INIT_GOOSE ────────────────────────────────────────────────
        if self.state == 'INIT_GOOSE':
            params = self.goa.get_individual(self.curr_goose_idx)
            self.ctrl.set_params(params)
            self.ctrl.reset()

            self.s           = 0.0
            self.v_s         = 0.02
            self.sum_es_sq   = 0.0
            self.count       = 0

            gen_disp = self.goa.generation + 1
            self.get_logger().info(
                f'Gen {gen_disp}/{self.goa.num_gens} | '
                f'Chạy ngỗng {self.curr_goose_idx + 1}/{self.goa.num_geese} ...'
            )
            self.state = 'RUNNING'
            return

        # ── RUNNING ───────────────────────────────────────────────────
        if self.state == 'RUNNING':

            # 1. Profile vận tốc
            v_adapt = self._v_safe(self.s)
            dist_brake = self.v_s**2 / (2 * self.a_path_max + 1e-9)

            if   (self.s_end - self.s) <= dist_brake: a_s = -self.a_path_max
            elif self.v_s < v_adapt - 0.01:           a_s =  self.a_path_max
            elif self.v_s > v_adapt + 0.01:           a_s = -self.a_path_max
            else:                                      a_s =  0.0

            self.v_s = float(np.clip(self.v_s + a_s * self.Ts, 0.01, self.v_path_max))
            self.s  += self.v_s * self.Ts

            # 2. Hết quỹ đạo → lưu fitness → đổi ngỗng
            if self.s >= self.s_end:
                rmse = math.sqrt(self.sum_es_sq / max(self.count, 1))
                self.goa.set_fitness(self.curr_goose_idx, rmse)

                self.get_logger().info(
                    f'   → Hoàn thành. RMSE = {rmse * 1000:.2f} mm'
                )

                self.curr_goose_idx += 1

                if self.curr_goose_idx >= self.goa.num_geese:
                    # ── Xong 1 thế hệ: cập nhật leader, tiến hóa ──
                    self.goa.update_leader()
                    self.get_logger().info(
                        f'\n{"="*50}\n'
                        f'🧬 KẾT THÚC THẾ HỆ {self.goa.generation + 1}/{self.goa.num_gens}\n'
                        f'🏆 {self.goa.summary()}\n'
                        f'{"="*50}\n'
                    )

                    if self.goa.is_done():
                        self.state = 'FINISHED'
                        return

                    # ── Tiến hóa sang thế hệ mới theo GOA chuẩn ──
                    self.goa.evolve()
                    self.curr_goose_idx = 0

                self._reset_gazebo_world()
                return

            # 3. Kinematics + điều khiển
            phi_ref, v_ref, w_ref = self.traj.ref_kinematics(self.s, self.v_s, a_s)
            x_ref, y_ref          = self.traj.pos(self.s)

            rx, ry, rphi = self.curr_q
            dx_g, dy_g   = x_ref - rx, y_ref - ry
            cq, sq       = math.cos(rphi), math.sin(rphi)
            e1 =  cq * dx_g + sq * dy_g
            e2 = -sq * dx_g + cq * dy_g
            e3 = math.atan2(math.sin(phi_ref - rphi), math.cos(phi_ref - rphi))

            v_cmd, w_cmd = self.ctrl.compute(v_ref, w_ref, e1, e2, e3)

            cmd = Twist()
            cmd.linear.x  = v_cmd
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)

            self.sum_es_sq += (dx_g**2 + dy_g**2)
            self.count     += 1


# ===========================================================================
def main():
    rclpy.init()
    node = Figure8GooseTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Dừng Auto-Tuner.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()