import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are, eigvals
from scipy import signal

##################################################################
# コントローラ・オブザーバ設計関連
# 可制御性判別用関数
def ctrb(A, B):
    """
    可制御性行列を計算
    A : システムの状態行列 (nxn 行列)
    B : 入力行列 (nxm 行列)
    controllability : 可制御性行列
    """
    n = A.shape[0]
    controllability = B

    for i in range(1, n):
        controllability = np.hstack((controllability, np.linalg.matrix_power(A, i) @ B))
    return controllability

# 可制御性判別結果表示
def is_controllable(A, B):
    """
    システムが可制御かどうかを判定し、結果を表示
    A : システムの状態行列 (nxn 行列)
    B : 入力行列 (nxm 行列)
    """
    ctr_matrix = ctrb(A, B)
    rank = np.linalg.matrix_rank(ctr_matrix)
    n = A.shape[0]
    print(f"Controllability Matrix:\n{ctr_matrix}")
    print(f"Rank of Controllability Matrix: {rank}")

# 可観測性判別用関数
def obsv(A, C):
    """
    可観測性行列を計算
    A : システムの状態行列 (nxn 行列)
    C : 観測行列 (pxn 行列)
    observability : 可観測性行列
    """
    n = A.shape[0]
    observability = C

    for i in range(1, n):
        observability = np.vstack((observability, C @ np.linalg.matrix_power(A, i)))
    return observability

# 可観測性判別結果表示
def is_observable(A, C):
    """
     システムが可観測かどうかを判定し、結果を表示
    A : システムの状態行列 (nxn 行列)
    C : 観測行列 (pxn 行列)
    """
    obs_matrix = obsv(A, C)
    rank = np.linalg.matrix_rank(obs_matrix)
    n = A.shape[0]
    print(f"Observability Matrix:\n{obs_matrix}")
    print(f"Rank of Observability Matrix: {rank}")

# 線形二次積分コントローラ設計
def lqi_design(A, B, C, Q, R):
    """
    線形二次積分 (LQI) コントローラを設計
    A : システムの状態行列
    B : 入力行列
    C : 観測行列
    Q : 状態重み行列
    R : 入力重み行列
    K : 状態フィードバックゲイン行列
    eig_ctrl_val : コントローラ適用後の閉ループ系の固有値
    """
    Ae = np.block([
        [A, np.zeros((A.shape[0], 1))],
        [-C, np.zeros((C.shape[0], 1))]
    ])
    Be = np.block([
        [B],
        [np.zeros((B.shape[1], 1))]
    ])
    Ce = np.block([
        [C, np.zeros((C.shape[0], 1))]
    ])

    P = solve_continuous_are(Ae, Be, Q, R)
    K = -np.linalg.inv(R).dot(Be.T).dot(P)

    print("Q : \r\n", Q)
    print("R : ", R)
    print("K : ", K)

    eig_ctrl_val = eigvals(Ae + Be @ K)
    print("eig_ctrl_val : \r\n", eig_ctrl_val)

    return K, eig_ctrl_val

# 極配置法
def place(A, C, p):
    """
    オブザーバゲインを極配置法で設計
    A : システムの状態行列
    C : 観測行列
    p : 配置する固有値のリスト
    L : オブザーバゲイン行列
    """
    place_result = signal.place_poles(A.T, C.T, p, method='YT')
    # オブザーバゲイン計算
    L = -place_result.gain_matrix.T
    # A+LCの極を計算
    eig_obsv_val, eig_obsv_vec = np.linalg.eig(A + L @ C)

    # 結果を表示
    print("p : ", p)
    print("L : \r\n", L)
    print("eig_obsv_val : ", eig_obsv_val)
    return L

# 定常カルマンフィルタ
def SteadyState_KalmanFilter(Ad, Cd, Q, R):
    """
    定常カルマンフィルタのゲインを計算
    Ad : 離散化されたシステムの状態行列
    Cd : 観測行列
    Q : システム雑音の共分散行列
    R : 観測雑音の共分散行列
    L : カルマンゲイン行列
    """
    P = solve_discrete_are(Ad.T, Cd.T, Q, R)
    L = -(Ad @ P @ Cd.T) @ np.linalg.inv(Cd @ P @ Cd.T + R)

    print("Qv : \r\n", Q)
    print("Rw : ", R)
    print("L : \r\n", L)
    return L

# 離散時間システムの固有値→連続時間システムの固有値に変換
def pole_dis2cont(dis_eig_val, dt):
    """
    離散時間系の固有値を連続時間系の固有値に変換
    dis_eig_val : 離散時間の固有値
    dt : サンプリング周期
    戻り値 : 連続時間系の固有値
    """
    return np.log(dis_eig_val) / dt

##################################################################
# シミュレーション関連
# 量子化器
class Quantizer:
    """
    入力信号を量子化するクラス
    """
    def __init__(self, resolution, val_max):
        """
        量子化器を初期化
        resolution: 分解能 (ステップ数)
        """
        self.resolution = resolution
        self.step_size = val_max / resolution # 1ステップあたりの値
    
    def get_step_size(self):
        return self.step_size
    
    def quantize(self, val):
        return np.round(val / self.step_size) * self.step_size

# 制御システム
class ControlSystemSimulator:
    """
    制御システムのシミュレータ。
    """
    def __init__(self, system_function, dt, dt_sim, A=None, B=None, C=None, K=None, L=None, control_mode="state_feedback", is_discrete=False):
        """
        制御システムシミュレータの初期化

        system_function : システムの運動方程式 (関数: f(x, u))
        dt : 制御周期
        dt_sim : シミュレーション周期
        A : システムの状態行列 (オブザーバに使用)、離散オブザーバを使うときは離散化したシステム行列を渡して
        B : 入力行列
        C : 観測行列
        K : ゲイン行列 (状態フィードバックまたはサーボ系で使用)
        L : オブザーバゲイン
        is_discrete : 離散時間システムかどうか
        """
        self.system_function = system_function
        self.dt = dt
        self.dt_sim = dt_sim
        self.A = A
        self.B = B
        self.C = C
        self.K = K
        self.L = L
        self.is_discrete = is_discrete
        self.integral_error = 0.0  # 偏差の積分値
        self.control_mode = control_mode  # デフォルトは通常の状態フィードバック
        self.vmax = None  # 入力制限値（デフォルトは制限なし）

    def set_control_mode(self, mode):
        """
        制御モードの切り替え
        mode: "state_feedback" または "servo"
        """
        if mode not in ["state_feedback", "servo"]:
            raise ValueError("Control mode should be either 'state_feedback' or 'servo'.")
        self.control_mode = mode

    def set_input_limit(self, vmax):
        """
        入力制限の設定
        vmax: 入力の最大絶対値
        """
        self.vmax = vmax

    def reset_integral_error(self):
        """
        偏差の積分値をリセット
        """
        self.integral_error = 0.0

    def runge_kutta_4th_step(self, x, u):
        """
        ルンゲクッタ法による1ステップの計算
        x: 状態ベクトル
        u: 制御入力
        """
        k1 = self.system_function(x, u)
        k2 = self.system_function(x + 0.5 * self.dt_sim * k1, u)
        k3 = self.system_function(x + 0.5 * self.dt_sim * k2, u)
        k4 = self.system_function(x + self.dt_sim * k3, u)
        return x + (self.dt_sim / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def state_feedback(self, x, x_ref=None):
        """
        状態フィードバック制御器
        x: 状態ベクトル
        x_ref: 目標値 (積分型最適サーボ系で使用)
        """
        if self.K is None:
            raise ValueError("Gain K is not set.")

        if self.control_mode == "state_feedback":
            # 通常の状態フィードバック
            u = self.K @ x
        elif self.control_mode == "servo":
            # 積分型最適サーボ系
            if x_ref is None:
                raise ValueError("The servo system requires a target value x_ref.")
            error = x_ref[0] - x[0]  # 偏差 (位置のみを考慮)
            self.integral_error += error * self.dt
            xe = np.block([[x[0], x[1], self.integral_error]]).T
            u = self.K @ xe

        # 入力制限を適用
        if self.vmax is not None:
            u = np.clip(u, -self.vmax, self.vmax)

        return u

    def state_observer(self, x_hat, y, u):
        """
        状態推定器の更新
        x_hat: 推定値
        y: 観測値
        u: 入力
        """
        if self.A is None or self.B is None or self.C is None or self.L is None:
            raise ValueError("System matrix or observer gain is not set.")

        ut = np.block([[u]])
        if self.is_discrete:
            # 離散時間オブザーバ
            x_hat = self.A @ x_hat + self.B @ ut - self.L @ (y - self.C @ x_hat)
        else:
            # 連続時間オブザーバ
            dx_hat = self.A @ x_hat + self.B @ ut - self.L @ (y - self.C @ x_hat)
            x_hat += dx_hat * self.dt
        return x_hat

##################################################################
# 制御系評価用
def analyze_response(time, response, target=1.0, tolerance=0.02):
    """
    システムのステップ応答を分析し、性能指標を計算

   
    time (np.ndarray): 時間配列
    response (np.ndarray): 応答の配列
    target (float): ステップ入力の目標値
    tolerance (float): 整定の許容値
    
    dict: 評価結果
    """
    # 許容範囲を計算
    lower_bound = target * (1 - tolerance)
    upper_bound = target * (1 + tolerance)

    # 最大行き過ぎ量
    overshoot = (np.max(response) - target) / target * 100

    # ピーク時間
    peak_index = np.argmax(response)
    peak_time = time[peak_index]

    # 立ち上がり時間 (10%～90%)
    rise_start_index = np.where(response >= 0.1 * target)[0][0]
    rise_end_index = np.where(response >= 0.9 * target)[0][0]
    rise_time = time[rise_end_index] - time[rise_start_index]

    # 整定時間を計算
    settling_indices = np.where((response >= lower_bound) & (response <= upper_bound))[0]
    if settling_indices.size > 0:
        # 最初に許容範囲に入ってから、最後まで範囲内に留まる時間を検出
        for idx in settling_indices:
            if np.all((response[idx:] >= lower_bound) & (response[idx:] <= upper_bound)):
                settling_time = time[idx]
                break
        else:
            settling_time = np.nan  # 整定しなかった場合
    else:
        settling_time = np.nan  # 範囲内に入らなかった場合

    # RMSE
    rmse = np.sqrt(np.mean((response - target) ** 2))

    # 結果をまとめる
    results = {
        "Peak Time [s]": peak_time,
        "Overshoot [%]": overshoot,
        "Rise Time [s]": rise_time,
        "Settling Time [s]": settling_time,
        "RMSE": rmse,
    }

    return results