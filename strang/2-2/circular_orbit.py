# p.113で紹介されている円軌道問題の数値解法の実装
# 4つの手法
# - 前進オイラー
# - 後退オイラー
# - 台形則
# - 蛙飛び法

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def neon_plot(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    (line,) = ax.plot(x, y, lw=1.0, zorder=6)
    for cont in range(6, 1, -1):
        ax.plot(x, y, lw=cont, color=line.get_color(), zorder=5, alpha=0.06)
    return line


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


def plot_orbits(methods, dt=2 * np.pi / 16, steps=64, show_exact=False):
    plt.rcParams.update(
        {
            "figure.facecolor": "#0b0f1a",
            "axes.facecolor": "#0b0f1a",
            "axes.edgecolor": "#6272a4",
            "axes.labelcolor": "#e6e6e6",
            "xtick.color": "#e6e6e6",
            "ytick.color": "#e6e6e6",
            "text.color": "#e6e6e6",
            "grid.color": "#6c6f93",
            "axes.prop_cycle": plt.cycler(
                "color", ["#00f5d4", "#f15bb5", "#9b5de5", "#fee440", "#00bbf9"]
            ),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    if show_exact:
        # 真の軌道（円）を描画
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            "--",
            color="#8be9fd",
            alpha=0.25,
            label="Exact (Circle)",
            zorder=2,
        )

    for method in methods:
        try:
            u_res, v_res = run_simulation(method, dt=dt, steps=steps)
            if u_res[1] is not None:
                line = neon_plot(u_res, v_res, ax=ax)
                line.set_label(method)
                ax.plot(
                    u_res,
                    v_res,
                    ".",
                    color=line.get_color(),
                    label="_nolegend_",
                    markersize=5,
                    zorder=7,
                )
        except TypeError:
            print(f"{method} はまだ実装されていません")

    ax.set_title("Phase Space Trajectories")
    ax.set_xlabel("Position u")
    ax.set_ylabel("Velocity v")
    ax.axis("equal")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.legend(
        frameon=False,
        labelcolor="#f5f5f5",
        handlelength=2.2,
        handletextpad=0.6,
    )
    ax.grid(True, alpha=0.25, zorder=1)
    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "circular_orbit.png"
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.show()


if __name__ == "__main__":
    # --- 実行と描画 ---
    methods = ["Forward Euler", "Backward Euler", "Trapezoidal", "Leapfrog"]
    plot_orbits(methods)
