from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import csv
import sys

import numpy as np

# ローカルimportを通すため、report直下を sys.path に追加する
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

from core.laplace_path_planning_solver import (
    SolveMethod,
    solve_laplace,
    trace_path_from_start,
)
from experiments.utils import build_problem_with_grid, default_layout_path, make_output_dir
from viz.plot_laplace import plot_laplace


def _choose_start_points(xlim: tuple[float, float], ylim: tuple[float, float]) -> list[tuple[float, float]]:
    x0, x1 = xlim
    y0, y1 = ylim
    dx = x1 - x0
    dy = y1 - y0
    return [
        (x0 + 0.75 * dx, y0 + 0.2 * dy),
        (x0 + 0.7 * dx, y0 + 0.5 * dy),
        (x0 + 0.8 * dx, y0 + 0.8 * dy),
    ]


def _snap_to_free_cell(
    start: tuple[float, float],
    occ: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[tuple[float, float], tuple[int, int]]:
    free = np.argwhere(~occ)
    if free.size == 0:
        raise ValueError("free cells not found in occupancy grid")
    sx, sy = start
    dx = xs[free[:, 0]] - sx
    dy = ys[free[:, 1]] - sy
    idx = int(np.argmin(dx * dx + dy * dy))
    i, j = free[idx]
    return (float(xs[i]), float(ys[j])), (int(i), int(j))


def main() -> int:
    # 入力レイアウトとソルバ設定
    layout_path = default_layout_path()
    output_dir = make_output_dir("exp_multi_start_points")
    omega = 1.5
    method = SolveMethod.SOR

    layout, problem, occ, xs, ys = build_problem_with_grid(layout_path)
    if layout.world.goal is None:
        raise ValueError("goal must be set in layout.yaml")

    result = solve_laplace(
        problem,
        method=method,
        omega=omega,
    )

    starts = _choose_start_points(layout.world.xlim, layout.world.ylim)
    snapped_starts: list[tuple[float, float]] = []
    paths: list[list[tuple[float, float]]] = []
    summaries = []

    for start in starts:
        snapped, _ = _snap_to_free_cell(start, occ, xs, ys)
        snapped_starts.append(snapped)
        path_result = trace_path_from_start(problem, result, start=snapped)
        path_xy = list(path_result.path_xy)
        paths.append(path_xy)
        last_x, last_y = path_xy[-1]
        i = int(np.argmin(np.abs(xs - last_x)))
        j = int(np.argmin(np.abs(ys - last_y)))
        reached = (i, j) in problem.goal_indices
        summaries.append(
            {
                "start_x": start[0],
                "start_y": start[1],
                "snapped_x": snapped[0],
                "snapped_y": snapped[1],
                "steps": len(path_xy),
                "reached_goal": reached,
            }
        )
        print(
            f"start=({start[0]:.2f}, {start[1]:.2f}) -> "
            f"snapped=({snapped[0]:.2f}, {snapped[1]:.2f}), "
            f"steps={len(path_xy)}, reached_goal={reached}"
        )

    ax_lin, ax_log = plot_laplace(occ, xs, ys, result.phi, goal=layout.world.goal)
    fig = ax_lin.get_figure()
    assert fig is not None
    fig.suptitle("Potential Field (multi-start)")

    colors = ["tab:orange", "tab:green", "tab:purple"]
    for idx, path_xy in enumerate(paths):
        px = [p[0] for p in path_xy]
        py = [p[1] for p in path_xy]
        label = f"path {idx + 1}"
        ax_lin.plot(px, py, color=colors[idx], linewidth=2.0, label=label)
        ax_log.plot(px, py, color=colors[idx], linewidth=2.0, label=label)

    for idx, start in enumerate(snapped_starts):
        ax_lin.scatter(
            start[0],
            start[1],
            color=colors[idx],
            s=30,
            label="_nolegend_",
        )
        ax_log.scatter(
            start[0],
            start[1],
            color=colors[idx],
            s=30,
            label="_nolegend_",
        )

    legend = ax_lin.get_legend()
    if legend is not None:
        legend.remove()
    handles, labels = ax_lin.get_legend_handles_labels()
    path_handles = []
    path_labels = []
    for handle, label in zip(handles, labels, strict=True):
        if label.startswith("path"):
            path_handles.append(handle)
            path_labels.append(label)
    fig.legend(
        path_handles,
        path_labels,
        loc="center left",
        bbox_to_anchor=(0.01, 0.5),
    )
    fig.tight_layout(rect=(0.12, 0.0, 1.0, 1.0))
    fig.savefig(output_dir / "multi_start_paths.png", dpi=200)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["start_x", "start_y", "snapped_x", "snapped_y", "steps", "reached_goal"]
        )
        for row in summaries:
            writer.writerow(
                [
                    f"{row['start_x']:.6f}",
                    f"{row['start_y']:.6f}",
                    f"{row['snapped_x']:.6f}",
                    f"{row['snapped_y']:.6f}",
                    row["steps"],
                    row["reached_goal"],
                ]
            )

    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
