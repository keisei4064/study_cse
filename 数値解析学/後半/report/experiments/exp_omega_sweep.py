from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys

# ローカルimportを通すため、report直下を sys.path に追加する
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

import numpy as np

from core.laplace_path_planning_solver import ProblemSpec, SolveMethod, solve_laplace
from core.path_planning_utils import goal_disk_indices
from problem_gen.occupancy_grid import load_layout_yaml, rasterize_occupancy_grid


def _build_problem(layout_path: Path) -> ProblemSpec:
    layout = load_layout_yaml(layout_path)
    if layout.world.goal is None:
        raise ValueError("goal must be set in layout.yaml")
    if layout.world.goal_radius is None:
        raise ValueError("goal_radius must be set in layout.yaml")

    occ, xs, ys = rasterize_occupancy_grid(layout)
    boundary_mask = np.zeros_like(occ, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    obstacle_mask = occ & ~boundary_mask

    wall_indices = {tuple(idx) for idx in np.argwhere(boundary_mask)}
    obstacle_indices = {tuple(idx) for idx in np.argwhere(obstacle_mask)}
    goal_indices = {
        (i, j)
        for (i, j) in goal_disk_indices(
            layout.world.goal,
            layout.world.goal_radius,
            xs,
            ys,
        )
        if not boundary_mask[i, j] and not obstacle_mask[i, j]
    }
    return ProblemSpec(
        xs=xs,
        ys=ys,
        wall_indices=wall_indices,
        obstacle_indices=obstacle_indices,
        goal_indices=goal_indices,
    )


def _default_omegas() -> list[float]:
    return [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]


def main() -> int:
    # 入力レイアウトと掃引設定
    layout_path = Path(__file__).resolve().parents[1] / "problem_gen" / "layout.yaml"
    output_dir = Path(__file__).resolve().parent / "exp_omega_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    omegas = _default_omegas()

    problem = _build_problem(layout_path)
    results = []
    for omega in omegas:
        result = solve_laplace(
            problem,
            method=SolveMethod.SOR,
            omega=omega,
        )
        results.append(result)

    # 収束履歴の比較
    import matplotlib.pyplot as plt

    fig0, ax0 = plt.subplots()
    for omega, result in zip(omegas, results, strict=True):
        ax0.plot(result.residual_norm_history, label=f"omega={omega:.2f}")
    ax0.set_yscale("log")
    ax0.set_xlabel("iteration")
    ax0.set_ylabel("residual (normalized)")
    ax0.set_title("SOR residual histories")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize="small", ncol=2)
    fig0.tight_layout()
    fig0.savefig(output_dir / "residual_histories.png", dpi=200)

    # 反復回数の比較
    fig1, ax1 = plt.subplots()
    iterations = [r.iterations for r in results]
    ax1.plot(omegas, iterations, marker="o")
    ax1.set_xlabel("omega")
    ax1.set_ylabel("iterations")
    ax1.set_title("SOR iterations vs omega")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(output_dir / "iterations_vs_omega.png", dpi=200)

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
