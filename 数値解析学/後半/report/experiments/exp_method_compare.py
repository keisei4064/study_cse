from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys
import csv

# ローカルimportを通すため、report直下を sys.path に追加する
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

from core.laplace_path_planning_solver import SolveMethod, solve_laplace
from experiments.utils import (
    build_problem_with_grid,
    default_layout_path,
    make_output_dir,
    measure_cpu_time,
)
from viz.plot_laplace import plot_laplace_log_only, plot_velocity_quiver_log


def _methods():
    return [
        ("Jacobi", SolveMethod.JACOBI, None),
        ("Gauss-Seidel", SolveMethod.GAUSS_SEIDEL, None),
        ("SOR (omega=1.5)", SolveMethod.SOR, 1.5),
    ]


def main() -> int:
    # 入力レイアウトと出力先
    layout_path = default_layout_path()
    output_dir = make_output_dir("exp_method_compare")

    layout, problem, occ, xs, ys = build_problem_with_grid(layout_path)
    results = []
    timings = []
    iterations = []

    for label, method, omega in _methods():
        if omega is None:
            result, cpu_time = measure_cpu_time(
                solve_laplace,
                problem,
                method=method,
            )
        else:
            result, cpu_time = measure_cpu_time(
                solve_laplace,
                problem,
                method=method,
                omega=omega,
            )
        results.append((label, result))
        timings.append((label, cpu_time))
        iterations.append((label, result.iterations))

    # 残差履歴の比較
    import matplotlib.pyplot as plt

    fig0, ax0 = plt.subplots()
    for label, result in results:
        ax0.plot(result.residual_norm_history, label=label)
    ax0.set_yscale("log")
    ax0.set_xlabel("iteration")
    ax0.set_ylabel(r"$\frac{\max |r_i|}{\max |r_0|}$", fontsize=15)
    ax0.set_title("Residual histories by method", pad=15)
    ax0.set_xlim(left=0)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize="small")
    fig0.tight_layout()
    fig0.savefig(output_dir / "residual_histories.png", dpi=200)

    # 反復回数の比較
    fig1, ax1 = plt.subplots(figsize=(5.5, 4))
    labels = [label for label, _ in iterations]
    iters = [value for _, value in iterations]
    ax1.bar(labels, iters, color="tab:blue", width=0.6)
    ax1.set_ylabel("iterations to convergence")
    ax1.set_title("Iterations to convergence by method", pad=15)
    ax1.grid(True, axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "iterations_by_method.png", dpi=200)

    # CPU time の比較
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    labels = [label for label, _ in timings]
    times = [value for _, value in timings]
    ax2.bar(labels, times, color="tab:orange", width=0.6)
    ax2.set_ylabel("CPU time [s]")
    ax2.set_title("CPU time by method", pad=15)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "cpu_time_by_method.png", dpi=200)

    # log可視化（2Dポテンシャル / 速度場）を横並びで一覧化
    fig_phi, axes_phi = plt.subplots(1, len(results), figsize=(13, 4), sharey=True)
    fig_vel, axes_vel = plt.subplots(1, len(results), figsize=(13, 4), sharey=True)
    if len(results) == 1:
        axes_phi = [axes_phi]
        axes_vel = [axes_vel]

    for ax, (label, result) in zip(axes_phi, results, strict=True):
        plot_laplace_log_only(
            occ,
            xs,
            ys,
            result.phi,
            start=layout.world.start,
            goal=layout.world.goal,
            ax=ax,
        )
        ax.set_title(label, pad=15)

    for ax, (label, result) in zip(axes_vel, results, strict=True):
        plot_velocity_quiver_log(occ, xs, ys, result.u, result.v, ax=ax)
        ax.set_title(label, pad=15)

    fig_phi.suptitle("Potential Field (2D, log)")
    fig_phi.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig_phi.subplots_adjust(wspace=0.15)
    fig_phi.savefig(output_dir / "potential_field_log.png", dpi=200)

    fig_vel.suptitle("Gradient Descent Flow (log)")
    fig_vel.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig_vel.subplots_adjust(wspace=0.15)
    fig_vel.savefig(output_dir / "gradient_descent_flow_log.png", dpi=200)

    # CSV 出力
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "iterations", "cpu_seconds"])
        for (label, iters), (_, cpu_time) in zip(iterations, timings, strict=True):
            writer.writerow([label, iters, f"{cpu_time:.6f}"])

    # コンソール出力
    for (label, iters), (_, cpu_time) in zip(iterations, timings, strict=True):
        print(f"{label}: iterations={iters}, cpu_time={cpu_time:.6f}s")

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
