# p.113で紹介されている円軌道問題の数値解法の実装
# 4つの手法
# - 前進オイラー
# - 後退オイラー
# - 台形則
# - 蛙飛び法

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from cycler import cycler


def run_simulation(
    method_name: str, dt: float = 2 * np.pi / 16, steps: int = 64
) -> tuple[np.ndarray, np.ndarray]:
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


def neon_plot(x: np.ndarray, y: np.ndarray, ax: Axes | None = None) -> Line2D:
    if ax is None:
        ax = plt.gca()
    (line,) = ax.plot(x, y, lw=1.0, zorder=6)
    for cont in range(7, 1, -1):
        ax.plot(x, y, lw=cont, color=line.get_color(), zorder=5, alpha=0.07)
    return line


def setup_neon_style() -> list[str]:
    color_cycle = ["#00f5d4", "#f15bb5", "#9b5de5", "#fee440", "#00bbf9"]
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
            "axes.prop_cycle": cycler("color", color_cycle),
        }
    )
    return color_cycle


def create_neon_lines(ax: Axes, color: str) -> tuple[Line2D, list[Line2D], Line2D]:
    glow_widths = [7, 6, 5, 4, 3]
    glow_lines = []
    for lw in glow_widths:
        (glow_line,) = ax.plot([], [], lw=lw, color=color, alpha=0.07, zorder=5)
        glow_lines.append(glow_line)
    (core_line,) = ax.plot([], [], lw=1.0, color=color, zorder=6)
    (point,) = ax.plot([], [], ".", color=color, markersize=6, zorder=7)
    return core_line, glow_lines, point


def plot_orbits(
    trajectories: Mapping[str, tuple[np.ndarray, np.ndarray]],
    show_exact: bool = False,
) -> None:
    setup_neon_style()

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

    for method, (u_res, v_res) in trajectories.items():
        try:
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
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim((-1.9, 1.9))
    ax.set_ylim((-1.9, 1.9))
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


def animate_orbits(
    trajectories: Mapping[str, tuple[np.ndarray, np.ndarray]],
    show_exact: bool = False,
    interval: int = 80,
    save_path: Path | None = None,
) -> None:
    color_cycle = setup_neon_style()
    fig, ax = plt.subplots(figsize=(10, 10))

    if show_exact:
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

    line_sets = {}

    methods = list(trajectories.keys())
    for method, color in zip(methods, (color_cycle * len(methods))):
        u_res, v_res = trajectories[method]
        core_line, glow_lines, point = create_neon_lines(ax, color)
        core_line.set_label(method)
        line_sets[method] = (core_line, glow_lines, point)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim((-1.9, 1.9))
    ax.set_ylim((-1.9, 1.9))
    ax.grid(True, alpha=0.25, zorder=1)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def init() -> list[Line2D]:
        artists: list[Line2D] = []
        for method in methods:
            core_line, glow_lines, point = line_sets[method]
            core_line.set_data([], [])
            point.set_data([], [])
            artists.append(core_line)
            artists.append(point)
            for glow_line in glow_lines:
                glow_line.set_data([], [])
                artists.append(glow_line)
        return artists

    def update(frame: int) -> list[Line2D]:
        artists: list[Line2D] = []
        for method in methods:
            u_res, v_res = trajectories[method]
            core_line, glow_lines, point = line_sets[method]
            x = u_res[: frame + 1]
            y = v_res[: frame + 1]
            core_line.set_data(x, y)
            point.set_data([u_res[frame]], [v_res[frame]])
            artists.append(core_line)
            artists.append(point)
            for glow_line in glow_lines:
                glow_line.set_data(x, y)
                artists.append(glow_line)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(next(iter(trajectories.values()))[0]),
        init_func=init,
        interval=interval,
        blit=True,
    )

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "circular_orbit.gif"
    anim.save(save_path, writer=animation.PillowWriter(fps=int(1000 / interval)))

    plt.show()


if __name__ == "__main__":
    # --- 実行と描画 ---
    methods = ["Forward Euler", "Backward Euler", "Trapezoidal", "Leapfrog"]
    trajectories = {method: run_simulation(method, steps=80) for method in methods}
    plot_orbits(trajectories)
    animate_orbits(trajectories)
