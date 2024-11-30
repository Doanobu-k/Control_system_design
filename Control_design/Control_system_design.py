import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_continuous_are, solve_discrete_are, eigvals
from scipy import signal
import ctrl_utils as ctrl
import sim_utils as sim

################################################
# シミュレーション条件
x_ref = np.array([[np.pi/2], [0]])      # 目標値
x0 = np.array([[0], [0]])               # 初期値
dt_sim = 1e-05                          # 時間ステップ
t_end = 3.0                             # シミュレーション終了時間
dt = 1e-03                              # 制御周期
vmax = 12                               # 最大入力電圧
delta_u = 2048                          # 入力の分解能
delta = 2400                            # エンコーダ分解能

# コントローラ設計用パラメータ
Q = np.diag([1.0e+05, 7.5e+02, 3.0e+07])    # LQIの状態変数に対する重み
R = np.array([[1e-00]])                     # LQIの入力に対する重み

# オブザーバ設計用パラメータ(極)
p = np.array([-300, -100])
# 定常カルマンフィルタ用パラメータ
Qv = np.array([
    [7.971e-02, -9.111e-04],
    [-9.111e-04, 3.388e+00]
])
Rw = np.array([[5.712e-07]])
################################################
# 状態方程式の定義
alpha = 25.6
beta = 16.3
gamma = 39.4
A = np.array([[0, 1],
              [0, -alpha]])
B = np.array([[0],
             [gamma]])
C = np.array([[1, 0]])

# 離散化
Ad, Bd, Cd, Dd, dt = signal.cont2discrete((A, B, C, np.zeros((1))), dt, method='zoh')
##################################################################
# 非線形システムの定義
def nonlinear_system(x, u):
    """
    非線形システムの運動方程式
    x : 状態ベクトル
    u : 制御入力
    """
    x1, x2 = x
    dx1 = x2
    dx2 = -alpha * x2 - beta * np.sign(x1) + gamma * u
    return np.array([dx1, dx2])  # 1次元配列として返す

##################################################################
# シミュレーション、プロット
def main():
    """
    シミュレーションの主処理
    1. 状態空間モデルと制御系の初期化
    2. 離散制御周期での制御計算
    3. シミュレーションステップごとのシステム更新
    4. 結果のプロット
    """

    ### 各種変数定義 ###
    t = np.arange(0, t_end, dt_sim)             # シミュレーション時間格納用バッファ
    x = np.zeros((len(x0), len(t)))             # 状態変数格納用バッファ
    u = np.zeros((B.shape[1], len(t)))          # 制御入力格納用バッファ
    u_actual = np.zeros((B.shape[1], len(t)))   # 実際の入力格納用バッファ
    x_hat = np.zeros((len(x0), len(t)))         # 推定値格納用バッファ
    u_ = np.zeros((B.shape[1], 1))              # 入力初期値
    x_hat_ = np.zeros((len(x0), 1))             # 推定値初期値
    x[:,0:1] = x0                               # 初期状態

    # コントローラゲイン・オブザーバゲイン計算
    Klqr, eig_ctrl_val = ctrl.lqi_design(A, B, C, Q, R)
    Lp = ctrl.place(A, C, p)
    Lk = ctrl.SteadyState_KalmanFilter(Ad, Cd, Qv, Rw)

    # システムの定義
    sys = ctrl.ControlSystemSimulator(nonlinear_system, dt, dt_sim, Ad, Bd, Cd, Klqr, Lk, control_mode="servo", is_discrete=True)
    # sys = ctrl.ControlSystemSimulator(nonlinear_system, dt, dt_sim, A, B, C, Klqr, Lp, control_mode="servo", is_discrete=False)
    sys.set_input_limit(vmax)

    # 量子化器
    enc_quantizer = ctrl.Quantizer(delta, 2*np.pi)
    u_quantizer = ctrl.Quantizer(delta_u, vmax)

    # ノイズ
    rng_obs = np.random.default_rng()
    rng_sys = np.random.default_rng()
    Qv_ = np.linalg.pinv(B) @ Qv @ np.linalg.pinv(B).T
    
    print(u_quantizer.get_step_size())
    
    jc = int(dt / dt_sim)   # 離散制御周期をシミュレーションステップでカウント

    ### シミュレーション ###
    for i in range(1, len(t)):
        # 離散システムのループ (制御周期ごとに実行)
        if jc == int(dt / dt_sim):
            # オブザーバ
            # 観測雑音生成
            w = rng_obs.uniform(-np.pi/2400, np.pi/2400+1e-10)
            y = C @ x[:,i-1] + w
            y = enc_quantizer.quantize(y)
            x_hat_ = sys.state_observer(x_hat_, y, u_)

            # コントローラ
            u_ = sys.state_feedback(x_hat_, x_ref)
            u_ = u_quantizer.quantize(u_)

            # 離散システム用カウントリセット
            jc = 0   
        
        # 観測雑音生成 (量子化器の分解能を考慮した一様分布)
        v = rng_sys.normal(0, (Qv_)**0.5)
        u_actual_ = u_ + v
        
        # 値保存
        u[:, i] = u_
        u_actual[:, i] = u_actual_
        x_hat[:,i] = x_hat_ .flatten()

        # システムのシミュレーション
        x[:, i] = sys.runge_kutta_4th_step(x[:, i-1], u_actual[0, i])
        # 離散システム用カウント
        jc += 1

    ### 結果を保存 ###
    headers = ["Time", "x", "dx", "x_hat", "dx_hat", "input", "input_actual"]
    data = np.column_stack((t, x.T, x_hat.T, u.T, u_actual.T))
    # sim.export_simulation_results_to_csv("simulation_results.csv", headers, data)

    ### 結果のプロット ###
    fig, ax = plt.subplots(1, 2, figsize=(12,5), squeeze=False)
    ax[0,0].axhline(x_ref[0,0], color='black', linestyle='--')  # 目標値 (角度)
    ax[0,0].axhline(0, color='black', linestyle='--')           # 基準線 (ゼロ)
    ax[0,0].set_xlim([0,t_end])
    ax[0,0].plot(t, x[0,:], label="Sim")
    ax[0,1].plot(t, x[1,:], label='Sim')
    ax[0,0].plot(t, x_hat[0,:], label="Obs")
    ax[0,1].plot(t, x_hat[1,:], label='Obs')
    ax[0,0].legend()
    ax[0,1].legend()

    fig1 = plt.figure()
    plt.plot(t, u_actual[0,:], label="u_actual")
    plt.plot(t, u[0,:], label="u")
    plt.show()

if __name__ == "__main__":
    main()