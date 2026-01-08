from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys
import csv
import numpy as np

# ローカルimportを通すため、report直下を sys.path に追加する
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

from core.laplace_path_planning_solver import SolveMethod, solve_laplace
from experiments.utils import (
    build_problem_with_grid_size,
    default_layout_path,
    make_output_dir,
    measure_cpu_time,
)
from viz.plot_laplace import (
    plot_laplace_log_only,
    plot_laplace_surface_3d_log,
    plot_velocity_quiver_log,
)


def _grid_sizes() -> list[int]:
    return [20, 40, 60, 80, 100, 120]


def main() -> int:
    # 入力レイアウトと出力先
    layout_path = default_layout_path()
    output_dir = make_output_dir("exp_resolution_sweep")

    sizes = _grid_sizes()
    iterations = []
    timings = []
    results = []
    grids = []

    for n in sizes:
        _, problem, occ, xs, ys = build_problem_with_grid_size(layout_path, n, n)
        result, cpu_time = measure_cpu_time(
            solve_laplace,
            problem,
            method=SolveMethod.SOR,
            omega=1.5,
        )
        iterations.append(result.iterations)
        timings.append(cpu_time)
        results.append(result)
        grids.append((occ, xs, ys))
        print(f"nx=ny={n}: iterations={result.iterations}, cpu_time={cpu_time:.6f}s")

    # 反復回数の比較
    import matplotlib.pyplot as plt

    fig0, ax0 = plt.subplots(figsize=(6, 4))
    ax0.bar([str(n) for n in sizes], iterations, color="tab:blue", width=0.6)
    ax0.set_xlabel("grid size (nx=ny)")
    ax0.set_ylabel("iterations to convergence")
    ax0.set_title("Iterations vs grid size", pad=15)
    ax0.grid(True, axis="y", alpha=0.3)
    fig0.tight_layout()
    fig0.savefig(output_dir / "iterations_vs_grid.png", dpi=200)

    # CPU time の比較
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar([str(n) for n in sizes], timings, color="tab:orange", width=0.6)
    ax1.set_xlabel("grid size (nx=ny)")
    ax1.set_ylabel("CPU time [s]")
    ax1.set_title("CPU time vs grid size", pad=15)
    ax1.grid(True, axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "cpu_time_vs_grid.png", dpi=200)

    # log可視化（2Dポテンシャル / 速度場）を横並びで一覧化
    ncols = 3
    nrows = int(np.ceil(len(sizes) / ncols))
    fig_phi, axes_phi = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), sharey=True
    )
    fig_vel, axes_vel = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), sharey=True
    )
    axes_phi = np.array(axes_phi, ndmin=2)
    axes_vel = np.array(axes_vel, ndmin=2)

    axes_phi_flat = axes_phi.ravel()
    axes_vel_flat = axes_vel.ravel()

    for idx, (n, result, (occ, xs, ys)) in enumerate(
        zip(sizes, results, grids, strict=True)
    ):
        ax = axes_phi_flat[idx]
        plot_laplace_log_only(
            occ,
            xs,
            ys,
            result.phi,
            ax=ax,
        )
        ax.set_title(f"n={n}", pad=15)

    for idx, (n, result, (occ, xs, ys)) in enumerate(
        zip(sizes, results, grids, strict=True)
    ):
        ax = axes_vel_flat[idx]
        plot_velocity_quiver_log(occ, xs, ys, result.u, result.v, ax=ax)
        ax.set_title(f"n={n}", pad=15)

    for ax in axes_phi_flat[len(sizes) :]:
        ax.axis("off")
    for ax in axes_vel_flat[len(sizes) :]:
        ax.axis("off")

    fig_phi.suptitle("Potential Field (2D, log)")
    fig_phi.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig_phi.savefig(output_dir / "potential_field_log.png", dpi=200)

    fig_vel.suptitle("Gradient Descent Flow (log)")
    fig_vel.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig_vel.savefig(output_dir / "gradient_descent_flow_log.png", dpi=200)

    fig_surface = plt.figure(figsize=(4.8 * ncols, 4.2 * nrows))
    axes_surface = [
        fig_surface.add_subplot(nrows, ncols, idx + 1, projection="3d")
        for idx in range(nrows * ncols)
    ]
    for idx, (n, result, (occ, xs, ys)) in enumerate(
        zip(sizes, results, grids, strict=True)
    ):
        ax = axes_surface[idx]
        plot_laplace_surface_3d_log(xs, ys, result.phi, occ=occ, ax=ax)
        ax.set_title(f"n={n}", pad=15)

    for ax in axes_surface[len(sizes) :]:
        ax.set_axis_off()

    fig_surface.suptitle("Potential Field (3D, log)")
    fig_surface.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig_surface.savefig(output_dir / "potential_field_3d_log.png", dpi=200)

    # CSV 出力
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["nx", "iterations", "cpu_seconds"])
        for n, iters, cpu_time in zip(sizes, iterations, timings, strict=True):
            writer.writerow([n, iters, f"{cpu_time:.6f}"])

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
