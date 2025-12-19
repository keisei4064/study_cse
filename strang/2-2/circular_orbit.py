# p.113で紹介されている円軌道問題の数値解法の実装
# 4つの手法
# - 前進オイラー
# - 後退オイラー
# - 台形則
# - 蛙飛び法

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def run_simulation(method_name, dt=2 * np.pi / 16, steps=64):
    # 初期条件: (u, v) = (1, 0) からスタート
    u = np.zeros(steps + 1)
    v = np.zeros(steps + 1)
    u[0] = 1.0
    v[0] = 0.0

    print(f"--- {method_name} の計算開始 ---")

    for n in range(steps):
        # 現在の値
        u_n = u[n]
        v_n = v[n]
        x_n = np.array([[u_n], [v_n]])
        x_n_plus_1 = np.zeros_like(x_n)

        # 次の値 (u_new, v_new) を計算
        if method_name == "Forward Euler":
            # 前進オイラー法 (Explicit)
            G_F = np.array(
                [
                    [1, dt],
                    [-dt, 1],
                ]
            )
            x_n_plus_1 = G_F @ x_n

        elif method_name == "Backward Euler":
            # 後退オイラー法 (Implicit)
            G_B = np.linalg.inv(
                np.array(
                    [
                        [1, -dt],
                        [dt, 1],
                    ]
                )
            )
            x_n_plus_1 = G_B @ x_n

        elif method_name == "Trapezoidal":
            # 台形則 (Trapezoidal)
            G_T = np.linalg.inv(
                np.array(
                    [
                        [1, -dt / 2],
                        [dt / 2, 1],
                    ]
                )
            ) @ np.array(
                [
                    [1, dt / 2],
                    [-dt / 2, 1],
                ]
            )
            x_n_plus_1 = G_T @ x_n

        elif method_name == "Leapfrog":
            # 蛙飛び法
            G_L = np.array(
                [
                    [1, dt],
                    [-dt, 1 - dt**2],
                ]
            )
            x_n_plus_1 = G_L @ x_n

        else:
            raise ValueError("Unknown method")

        # ---------------------------------------------------------

        # 配列に格納
        u[n + 1] = x_n_plus_1[0, 0]
        v[n + 1] = x_n_plus_1[1, 0]

    return u, v


# --- 実行と描画 ---
methods = ["Forward Euler", "Backward Euler", "Trapezoidal", "Leapfrog"]
plt.figure(figsize=(10, 10))

# 真の軌道（円）を描画
# theta = np.linspace(0, 2 * np.pi, 100)
# plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Exact (Circle)")

for method in methods:
    try:
        u_res, v_res = run_simulation(method)
        # 成功したらプロット (Noneのままだとエラーになります)
        if u_res[1] is not None:
            plt.plot(u_res, v_res, ".-", label=method, markersize=4)
    except TypeError:
        print(f"{method} はまだ実装されていません")

plt.title("Phase Space Trajectories")
plt.xlabel("Position u")
plt.ylabel("Velocity v")
plt.axis("equal")
plt.xlim((-2, 2))
plt.ylim((-2, 2))
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path = Path(__file__).resolve().parent / "circular_orbit.png"
plt.savefig(output_path, dpi=150)
plt.show()
