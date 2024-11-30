import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import csv

### 高速フーリエ変換 ###
def FFT(data, fs):
    N = len(data)
    window = signal.hann(N)
    F = np.fft.fft(data * window)
    freq = np.fft.fftfreq(N, d=1/fs)    # 周波数スケール
    F = F / (N / 2)                     # フーリエ変換の結果を正規化
    F = F * (N / sum(window))           # 窓関数による振幅減少を補正する
    Amp = np.abs(F)                     # 振幅スペクトル
    return Amp, freq

def FFT_plot_set(Amp, freq, N, labelname, ff=None):
    if ff is None:
        ff = plt.figure()
    else:
        plt.figure(ff.number)
    
    plt.plot(freq[:N//2], Amp[:N//2], label=labelname, linewidth=2)
    plt.xlabel("Frequency [Hz]", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.xlim(0,None)
    plt.ylim(0,None)
    plt.tick_params(labelsize=12)
    plt.grid()
    plt.legend()


### ローパスフィルタ用関数 ###
def butter_lowpass_filter(x, lowcut, fs, order=4):
    nyq = 0.5 *fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype='low')
    y = signal.filtfilt(b, a, x)
    return y

### ハイパスフィルタ用関数 ###
def butter_highpass_filter(x, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, highcut, btype='high')
    y = signal.filtfilt(b, a, x)
    return y

### 中心差分近似 ###
def central_diff(x, dt):
    k = len(x)
    h = dt
    dx = np.zeros(k)
    dx[0] = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * h)
    for i in range (1, k-1):
        dx[i] = (x[i+1] - x[i-1]) / (2 * h)
    dx[k-1] = (x[k-3] - 4 * x[k-2] + 3 * x[k-1]) / (2 * h)
    return dx

def central_diff_dynamic_time(x, t):
    """
    中心差分近似を与えられた時刻データから動的に計算します。
    
    Parameters:
    x : array-like
        データ点の配列。
    t : array-like
        各データ点に対応する時刻の配列（xと同じ長さ）。
        
    Returns:
    dx : np.ndarray
        中心差分で計算された微分値の配列。
    """
    k = len(x)
    dx = np.zeros(k)
    
    # 時刻間隔を計算
    dt = np.diff(t)  # 各区間の時間差 (長さ k-1)
    
    # 前進差分で計算（最初の点）
    dx[0] = (-3 * x[0] + 4 * x[1] - x[2]) / (2 * dt[0])
    
    # 中心差分で計算（内部の点）
    for i in range(1, k-1):
        h1 = t[i] - t[i-1]  # 前の間隔
        h2 = t[i+1] - t[i]  # 次の間隔
        dx[i] = (x[i+1] - x[i-1]) / (h1 + h2)
    
    # 後退差分で計算（最後の点）
    dx[k-1] = (x[k-3] - 4 * x[k-2] + 3 * x[k-1]) / (2 * dt[-1])
    
    return dx

### オイラー法 ###
def euler_method(t, dt, x0, v0, func, u):
    dx = np.zeros(len(t))
    x = np.zeros(len(t))
    for i in range(len(t) - 1):
        if i == 0:
            dx[i] = v0
            x[i] = x0
        dx[i + 1] = dx[i] + func(x[i], dx[i], u[i]) * dt
        x[i + 1] = x[i] + dx[i] * dt
    return x, dx

### ルンゲクッタ法 ###
def runge_kutta_method(t, dt, x0, v0, func, u):
    ddx = np.zeros(len(t))
    dx = np.zeros(len(t))
    x = np.zeros(len(t))
    for i in range(len(t) - 1):
        if i == 0:
            dx[i] = v0
            x[i] = x0
        k1_x = dx[i] * dt
        k1_v = func(x[i], dx[i], u[i]) * dt
        k2_x = (dx[i] + k1_v / 2) * dt
        k2_v = func(x[i] + k1_x / 2, dx[i] + k1_v / 2, u[i]) * dt
        k3_x = (dx[i] + k2_v / 2) * dt
        k3_v = func(x[i] + k2_x / 2, dx[i] + k2_v / 2, u[i]) * dt
        k4_x = (dx[i] + k3_v) * dt
        k4_v = func(x[i] + k3_x, dx[i] + k3_v, u[i]) * dt

        ddx[i + 1] = (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / (6 * dt)
        dx[i + 1] = dx[i] + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        x[i + 1] = x[i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    return x, dx, ddx

# 運動方程式 上記のfuncはこんな感じで運動方程式を関数として定義して使って
# def func(x, dx, u, a = p[0], b = p[1], c = p[2]):
#     return - a * dx - b * np.sign(dx) + c * u

### 適合率(システム同定の評価) ###
def fit_ratio(y_actual, y_model):
    """
    適合率を計算する関数
    Parameters:
    y_actual (np.array): 実測データの出力ベクトル
    y_model (np.array): 同定したモデルの出力ベクトル
    Returns:
    float: 適合率（百分率）
    """
    # 実測データの平均値
    y_mean = np.mean(y_actual)

    # 分母：実測データとその平均値との差の二乗和の平方根
    denominator = np.linalg.norm(y_actual - y_mean)
    # 分子：実測データとモデル出力の差の二乗和の平方根
    numerator = np.linalg.norm(y_actual - y_model)

    # 適合率の計算
    fit = (1 - (numerator / denominator)) * 100
    return fit

### M系列信号生成 ###
def generate_m_sequence(n, taps):
    """
    M系列信号を生成する関数
    n: レジスタのビット数
    taps: フィードバックタップの位置
    """
    # 初期状態
    state = [1] * n
    m_seq = []
    
    for _ in range(2**n - 1):
        # 出力ビット
        output = state[-1]
        m_seq.append(output)
        
        # 新しいフィードバックビットの計算
        feedback = sum(state[tap - 1] for tap in taps) % 2
        
        # シフト操作
        state = [feedback] + state[:-1]
    
    return np.array(m_seq)

# シミュレーション結果をCSVに出力する関数
def export_simulation_results_to_csv(filename, headers, data):
    """
    シミュレーション結果をCSVファイルに出力する関数
    filename (str) : 出力するCSVファイル名
    headers (list) : CSVのヘッダー行（列名）
    data (list or np.ndarray) : シミュレーション結果のデータ (行ごとに記録される)
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("Data should be provided in list or NumPy array format.")

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # ヘッダーを書き込む
        if headers:
            writer.writerow(headers)

        # データを書き込む
        for row in data:
            writer.writerow(row)

    print(f"Simulation results output to CSV file '{filename}'.")