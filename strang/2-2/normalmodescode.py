# p.116のサンプルコード

from pathlib import Path

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import animation


def compute_normal_modes():
    # rosscode.m のパラメータを使用
    M = np.array(
        [
            [9.0, 0.0],
            [0.0, 1.0],
        ]
    )
    K = np.array(
        [
            [81.0, -6.0],
            [-6.0, 6.0],
        ]
    )

    # 初期条件
    u_zero = np.array([1.0, 0.0])
    v_zero = np.array([0.0, 0.0])

    # 1. 一般化固有値問題を解く Kx = lambda Mx
    #   scipy.linalg.eigh は "Hermitian" (対称行列) 用のソルバ
    #   type=1 は A x = w B x 形式を指定
    evals, evecs = la.eigh(K, M, type=1)
    #   evals: 固有値 (lambda = omega^2)
    #   evecs: 固有ベクトル行列 (M直交している)

    # 固有振動数 ω = sqrt(λ)
    omega = np.sqrt(evals)

    print(f"固有値 (lambda): {evals}")
    print(f"固有振動数 (omega): {omega}")
    print(f"固有ベクトル:\n{evecs}")

    # 2. 係数 A, B を求める
    #       u(0) = sum(A_i * x_i),
    #       v(0) = sum(B_i * omega_i * x_i)
    #   行列形式だと:
    #       u_zero = V * A_coeffs,
    #       v_zero = V * (omega * B_coeffs)

    # Pythonの solve は "vectors \ uzero" に相当
    A_coeffs = la.solve(evecs, u_zero)

    # Bは今回 v_zero=0 なので全ゼロだが、一般形として記述
    B_coeffs = la.solve(evecs, v_zero) / omega

    # 3. 時間発展を計算
    t = np.linspace(0, 10, 500)

    # 各モードの振動を足し合わせる
    #   coeffs(t) = A cos(wt) + B sin(wt)
    coeffs = A_coeffs[:, None] * np.cos(omega[:, None] * t) + B_coeffs[
        :, None
    ] * np.sin(omega[:, None] * t)

    # u(t) = Matrix_V @ vector_coeffs(t)
    u_t = evecs @ coeffs

    return u_t, t


def plot_normal_modes(u_t, t, output_dir):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t, u_t[0, :], label="u1 (Mass 9)", color="green")
    plt.plot(t, u_t[1, :], label="u2 (Mass 1)", color="hotpink")  # 第2章カラー
    plt.title("Normal Modes Solution (Exact)")
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    output_path = output_dir / "normal_modes.png"
    fig.savefig(output_path, dpi=150)
    plt.show()
    return fig


def animate_spring_mass(u_t, t, output_dir):
    # 上端固定の2自由度ばね質点系のアニメーション
    u1 = u_t[0, :]
    u2 = u_t[1, :]

    # 平衡位置（任意の見た目スケール）
    y_top = 0.0
    y1_eq = 1.3
    y2_eq = 5.0

    y1 = y1_eq + u1
    y2 = y2_eq + u2

    y_min = min(y_top, np.min(y1), np.min(y2)) - 0.5
    y_max = max(y_top, np.max(y1), np.max(y2)) + 0.5

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(y_max, y_min)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Spring-Mass Animation (Top Fixed)")

    # 固定端とばね・質点
    (top_point,) = ax.plot([0], [y_top], "ks", markersize=8)
    (spring1,) = ax.plot([], [], color="gray", linewidth=2)
    (spring2,) = ax.plot([], [], color="gray", linewidth=2)
    (mass1,) = ax.plot([], [], "o", color="green", markersize=14)
    (mass2,) = ax.plot([], [], "o", color="hotpink", markersize=8)
    time_text = ax.text(
        0.05,
        0.05,
        "",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    def init():
        spring1.set_data([], [])
        spring2.set_data([], [])
        mass1.set_data([], [])
        mass2.set_data([], [])
        time_text.set_text("")
        return spring1, spring2, mass1, mass2, time_text

    def update(frame):
        y1f = y1[frame]
        y2f = y2[frame]
        spring1.set_data([0, 0], [y_top, y1f])
        spring2.set_data([0, 0], [y1f, y2f])
        mass1.set_data([0], [y1f])
        mass2.set_data([0], [y2f])
        time_text.set_text(f"t = {t[frame]:.2f}")
        return spring1, spring2, mass1, mass2, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=20,
        blit=True,
    )
    output_path = output_dir / "spring_mass.gif"
    anim.save(output_path, writer=animation.PillowWriter(fps=30))
    plt.show()
    return anim


def main():
    output_dir = Path(__file__).resolve().parent
    u_t, t = compute_normal_modes()
    plot_normal_modes(u_t, t, output_dir)
    animate_spring_mass(u_t, t, output_dir)


if __name__ == "__main__":
    main()
