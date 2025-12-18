# p.113で紹介されている円軌道問題の数値解法の実装
# 4つの手法
# - 前進オイラー
# - 後退オイラー
# - 台形則
# - 蛙飛び法

import numpy as np
import matplotlib.pyplot as plt


def run_simulation(method_name, dt=2 * np.pi / 32, steps=100):
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

        # 次の値 (u_new, v_new) を計算する部分
        # ---------------------------------------------------------
        # 【ここを埋めてください】
        # ヒント: u_n, v_n, dt を使って u_new, v_new を表現します
        # ---------------------------------------------------------

        if method_name == "Forward Euler":
            # 前進オイラー法 (Explicit)
            # u_{n+1} = u_n + dt * v_n
            # v_{n+1} = v_n - dt * u_n
            u_new = None  # ???
            v_new = None  # ???

        elif method_name == "Backward Euler":
            # 後退オイラー法 (Implicit)
            # 行列を解いた結果の式をここに書く必要があります
            # ((1, -dt), (dt, 1)) の逆行列的な係数がかかります
            denom = 1 + dt**2
            u_new = None  # ???
            v_new = None  # ???

        elif method_name == "Symplectic Euler":
            # シンプレクティックオイラー (Semi-implicit)
            # 片方を新しい値、もう片方を古い値で更新します
            # 例: 先に u を更新し、その新しい u を使って v を更新
            u_new = None  # ???
            v_new = None  # ??? (ここでは u_new を使ってもよい)

        elif method_name == "Leapfrog":
            # 蛙飛び法 (またはそれに類する2次精度のもの)
            # ※ここでは、教科書でよく比較される「半ステップずらす」概念などを
            #   簡易的に実装するか、Trapezoidal (台形公式) を入れることが多いです。
            #   一旦、台形公式 (Trapezoidal) として枠を用意しておきます。
            #   (1 - dt^2/4) / (1 + dt^2/4) のような係数が出てきます
            decay = (1 - dt**2 / 4) / (1 + dt**2 / 4)
            # ...台形公式は式が少し複雑なので、まずは上の3つに集中してもOKです
            u_new = u_n  # 仮置き
            v_new = v_n  # 仮置き

        else:
            raise ValueError("Unknown method")

        # ---------------------------------------------------------

        # 配列に格納
        u[n + 1] = u_new
        v[n + 1] = v_new

    return u, v


# --- 実行と描画 ---
methods = ["Forward Euler", "Backward Euler", "Symplectic Euler", "Leapfrog"]
plt.figure(figsize=(10, 10))

# 真の軌道（円）を描画
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Exact (Circle)")

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
plt.legend()
plt.grid(True)
plt.show()
