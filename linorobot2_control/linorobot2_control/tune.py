#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
import optuna
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
import time
from collections import deque

# ============ CONFIG ============
ROBOT_NAME = "differential_drive_robot"
CMD_VEL_TOPIC = "/cmd_vel"
ODOM_TOPIC = "/odom"
WHEELBASE = 0.5  # Khoảng cách 2 bánh (m)
SIMULATION_TIME = 15  # Thời gian chạy mỗi trial (s)
N_TRIALS = 50  # Số lần tối ưu
TRAJECTORY_RADIUS = 2.0  # Bán kính quỹ đạo tròn (m)

# ============ CIRCULAR TRAJECTORY ============
def create_circular_trajectory(radius=2.0, duration=15, dt=0.1):
    """Tạo quỹ đạo tròn"""
    t = np.arange(0, duration, dt)
    angular_vel = 0.5  # rad/s
    
    angles = angular_vel * t
    
    x_ref = radius * np.cos(angles)
    y_ref = radius * np.sin(angles)
    theta_ref = angles + np.pi/2  # Hướng tiếp tuyến
    
    v_ref = np.full_like(t, radius * angular_vel)  # ✅ FIX: Array, not float
    
    return t, x_ref, y_ref, theta_ref, v_ref

# ============ LQR CONTROLLER ============
class LQRController:
    def __init__(self, wheelbase=0.5, dt=0.1):
        self.wheelbase = wheelbase
        self.dt = dt
        
    def compute_lqr_gains(self, Q, R):
        """Tính ma trận K từ Q, R"""
        # Mô hình tuyến tính hóa: x_error, y_error, theta_error
        A = np.array([
            [0, 0, -0.5],
            [0, 0, 0.5],
            [0, 0, 0]
        ])
        
        B = np.array([
            [0.5/self.wheelbase, -0.5/self.wheelbase],
            [0.5/self.wheelbase, -0.5/self.wheelbase],
            [1/self.wheelbase, -1/self.wheelbase]
        ])
        
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except:
            return np.zeros((2, 3))
    
    def control_law(self, state_error, v_ref, omega_ref, K):
        """Tính điều khiển LQR"""
        # Điều khiển tham chiếu
        v_left_ref = v_ref - omega_ref * self.wheelbase / 2
        v_right_ref = v_ref + omega_ref * self.wheelbase / 2
        
        # Sửa lỗi từ LQR
        u_correction = K @ state_error
        
        # Điều khiển cuối cùng
        v_left = v_left_ref + u_correction[0]
        v_right = v_right_ref + u_correction[1]
        
        # Saturate
        v_left = np.clip(v_left, -1.0, 1.0)
        v_right = np.clip(v_right, -1.0, 1.0)
        
        return v_left, v_right

# ============ ROS2 NODE ============
class GazeboLQROptimizer(Node):
    def __init__(self):
        super().__init__('gazebo_lqr_optimizer')
        
        # Publisher & Subscriber
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10)
        
        # Service client cho reset
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        
        # Odometry
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        
        # History
        self.x_history = deque(maxlen=1000)
        self.y_history = deque(maxlen=1000)
        self.theta_history = deque(maxlen=1000)
        
        self.get_logger().info("✅ GazeboLQROptimizer started!")
    
    def odom_callback(self, msg):
        """Nhận odometry từ Gazebo"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Lấy theta từ quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        
        self.current_theta = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        
        self.x_history.append(self.current_x)
        self.y_history.append(self.current_y)
        self.theta_history.append(self.current_theta)
    
    def publish_cmd_vel(self, v_left, v_right):
        """Gửi lệnh điều khiển"""
        msg = Twist()
        msg.linear.x = (v_left + v_right) / 2
        msg.angular.z = (v_right - v_left) / WHEELBASE
        
        self.cmd_vel_pub.publish(msg)
    
    def reset_simulation(self):
        """Reset Gazebo"""
        req = Empty.Request()
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Chờ /reset_world service...')
        
        future = self.reset_world_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("🔄 Reset thành công!")
        
        self.x_history.clear()
        self.y_history.clear()
        self.theta_history.clear()
        time.sleep(1)
    
    def run_lqr_trial(self, Q, R):
        """Chạy 1 trial LQR"""
        self.get_logger().info(f"🚀 Chạy trial: Q={np.diag(Q)}, R={np.diag(R)}")
        
        # Reset
        self.reset_simulation()
        
        # Tạo quỹ đạo
        t, x_ref, y_ref, theta_ref, v_ref = create_circular_trajectory(
            radius=TRAJECTORY_RADIUS,
            duration=SIMULATION_TIME,
            dt=0.1
        )
        
        # Tính K
        lqr = LQRController(wheelbase=WHEELBASE, dt=0.1)
        K = lqr.compute_lqr_gains(Q, R)
        
        # Chạy
        x_actual, y_actual, theta_actual = [], [], []
        errors = []
        
        start_time = time.time()
        
        for i, time_step in enumerate(t):
            # Lỗi hiện tại
            x_error = self.current_x - x_ref[i]
            y_error = self.current_y - y_ref[i]
            theta_error = self.current_theta - theta_ref[i]
            
            state_error = np.array([x_error, y_error, theta_error])
            
            # Tính điều khiển
            omega_ref = 0.5  # rad/s (vòng tròn)
            v_left, v_right = lqr.control_law(state_error, v_ref[i], omega_ref, K)
            
            # Gửi lệnh
            self.publish_cmd_vel(v_left, v_right)
            
            # Lưu
            x_actual.append(self.current_x)
            y_actual.append(self.current_y)
            theta_actual.append(self.current_theta)
            
            # Tính lỗi
            pos_error = np.sqrt(x_error**2 + y_error**2)
            errors.append(pos_error)
            
            # Chờ
            time.sleep(0.1)
            rclpy.spin_once(self, timeout_sec=0.01)
            
            # Kiểm tra timeout
            if time.time() - start_time > SIMULATION_TIME + 5:
                break
        
        # Tính lỗi tổng
        if len(errors) > 0:
            rmse = np.sqrt(np.mean(np.array(errors)**2))
        else:
            rmse = 1000.0
        
        self.get_logger().info(f"✅ Trial xong! RMSE = {rmse:.4f}")
        
        return rmse, x_actual, y_actual, theta_actual, x_ref, y_ref

# ============ OPTUNA OBJECTIVE ============
def create_objective(node, x_ref, y_ref, theta_ref, v_ref):
    def objective(trial):
        # Gợi ý Q, R
        q1 = trial.suggest_float('q1', 0.1, 100.0)
        q2 = trial.suggest_float('q2', 0.1, 100.0)
        q3 = trial.suggest_float('q3', 0.1, 100.0)
        r1 = trial.suggest_float('r1', 0.01, 10.0)
        r2 = trial.suggest_float('r2', 0.01, 10.0)
        
        Q = np.diag([q1, q2, q3])
        R = np.diag([r1, r2])
        
        # Chạy trial
        rmse, _, _, _, _, _ = node.run_lqr_trial(Q, R)
        
        return rmse
    
    return objective

# ============ MAIN ============
def main(args=None):
    rclpy.init(args=args)
    
    node = GazeboLQROptimizer()
    
    # Chờ Gazebo ready
    time.sleep(2)
    
    # Tạo quỹ đạo tham chiếu
    t, x_ref, y_ref, theta_ref, v_ref = create_circular_trajectory(
        radius=TRAJECTORY_RADIUS,
        duration=SIMULATION_TIME,
        dt=0.1
    )
    
    # Optuna
    print("\n" + "="*60)
    print("🔍 OPTUNA LQR TUNING - GAZEBO")
    print("="*60)
    
    objective = create_objective(node, x_ref, y_ref, theta_ref, v_ref)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Kết quả
    print("\n" + "="*60)
    print("🏆 KẾT QUẢ TỐI ƯU")
    print("="*60)
    print(f"Lỗi tốt nhất: {study.best_value:.4f}")
    print(f"Tham số tốt nhất:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # Chạy lại lần cuối với tham số tốt nhất
    print("\n🎬 Chạy lại lần cuối với tham số tốt nhất...")
    
    q1 = study.best_params['q1']
    q2 = study.best_params['q2']
    q3 = study.best_params['q3']
    r1 = study.best_params['r1']
    r2 = study.best_params['r2']
    
    Q_best = np.diag([q1, q2, q3])
    R_best = np.diag([r1, r2])
    
    rmse_best, x_actual, y_actual, theta_actual, x_ref, y_ref = node.run_lqr_trial(Q_best, R_best)
    
    # Vẽ đồ thị
    print("\n📊 Vẽ đồ thị...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Quỹ đạo
    axes[0, 0].plot(x_ref, y_ref, 'g--', label='Reference', linewidth=2)
    axes[0, 0].plot(x_actual, y_actual, 'b-', label='Actual', linewidth=1.5)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Trajectory Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid()
    axes[0, 0].axis('equal')
    
    # Lỗi vị trí
    errors = np.sqrt(np.array(x_actual)**2 + np.array(y_actual)**2 - 
                     np.array(x_ref)**2 - np.array(y_ref)**2)
    axes[0, 1].plot(errors, 'r-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Position Error (m)')
    axes[0, 1].set_title(f'Tracking Error (RMSE={rmse_best:.4f})')
    axes[0, 1].grid()
    
    # Trial history
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    axes[1, 0].plot(trial_values, 'o-', color='purple', linewidth=1.5, markersize=4)
    axes[1, 0].axhline(y=study.best_value, color='r', linestyle='--', label='Best')
    axes[1, 0].set_xlabel('Trial Number')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Optimization Progress')
    axes[1, 0].legend()
    axes[1, 0].grid()
    
    # Q, R parameters
    axes[1, 1].bar(range(5), [q1, q2, q3, r1, r2], color=['blue', 'blue', 'blue', 'red', 'red'])
    axes[1, 1].set_xticks(range(5))
    axes[1, 1].set_xticklabels(['Q1', 'Q2', 'Q3', 'R1', 'R2'])
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Best Parameters')
    axes[1, 1].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('lqr_optimization_results.png', dpi=150)
    print("✅ Lưu đồ thị: lqr_optimization_results.png")
    plt.show()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()