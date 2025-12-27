from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. 対象の微分方程式を定義
#    数式: u' = -100u + 100sin(t)
# ---------------------------------------------------------
def problem4_ode(t, u):
    dudt = -100 * u + 100 * np.sin(t)
    return dudt


# ---------------------------------------------------------
# 2. 4次ルンゲ・クッタ法 (RK4) の1ステップ
#    教科書 p.470 式(27)
# ---------------------------------------------------------
def rk4_step(func, t, u, dt):
    k1 = func(t, u)

    # k2: 半歩先 (0.5*dt) での傾き。u は k1 の勢いで進む
    k2 = func(t + 0.5 * dt, u + 0.5 * dt * k1)

    # k3: 半歩先での傾き。ただし u は k2 の勢いで進む修正版
    k3 = func(t + 0.5 * dt, u + 0.5 * dt * k2)

    # k4: 1歩先 (dt) での傾き。u は k3 の勢いで進む
    k4 = func(t + dt, u + dt * k3)

    # 最後に加重平均をとって進む
    # 係数は 1 : 2 : 2 : 1
    u_next = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return u_next


# ---------------------------------------------------------
# シミュレーション実行部分
# ---------------------------------------------------------
def run_simulation(dt, label):
    t_span = [0, 10]  # 0秒から3秒まで
    t_values = np.arange(t_span[0], t_span[1], dt)
    u_values = np.zeros(len(t_values))

    # 初期値 u(0) = 0
    u_values[0] = 0.0

    print(f"Simulation start: dt={dt}")

    for i in range(len(t_values) - 1):
        t = t_values[i]
        u = u_values[i]

        # RK4で次の値を計算
        u_next = rk4_step(problem4_ode, t, u, dt)

        # 発散チェック (値が大きすぎたら計算打ち切り)
        if abs(u_next) > 1e6:
            print(f"  -> BLOWN UP at t={t:.4f}!")
            u_values[i + 1 :] = np.nan
            break

        u_values[i + 1] = u_next

    plt.plot(t_values, u_values, label=label)


# メイン処理
fig = plt.figure(figsize=(10, 6))

# 2つのステップ幅で実験
# 安定限界 -2.78 / -100 = 0.0278 付近をテスト
run_simulation(dt=0.0275, label="dt=0.0275 (Stable)")
run_simulation(dt=0.0280, label="dt=0.0280 (Unstable)")
run_simulation(dt=0.0300, label="dt=0.0300 (Unstable)")

plt.title("RK4 Stability Test on Stiff ODE (stability limit ~ 0.0278)")
plt.xlabel("Time t")
plt.ylabel("u(t)")
plt.ylim(-2, 2)  # 正常ならこの範囲に収まるはず
plt.legend()
plt.grid(True)
output_path = Path(__file__).with_name("rk4_stiff_stability.png")
fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
plt.show()
