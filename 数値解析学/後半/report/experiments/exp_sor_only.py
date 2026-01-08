from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys

# Ensure the report root is on sys.path for local imports.
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

import numpy as np

from core.laplace_path_planning_solver import (
    ProblemSpec,
    SolveMethod,
    solve_laplace,
    trace_path_from_start,
)
from core.path_planning_utils import goal_disk_indices
from problem_gen.occupancy_grid import load_layout_yaml, rasterize_occupancy_grid
from viz.plot_laplace import (
    plot_laplace,
    plot_laplace_surface_3d_pair,
    plot_velocity_quiver_pair,
    plot_residual_history,
)


def main() -> int:
    layout_path = Path(__file__).resolve().parents[1] / "problem_gen" / "layout.yaml"
    omega = 1.5
    max_iter = 10_000
    tol = 1.0e-5
    method = SolveMethod.SOR

    layout = load_layout_yaml(layout_path)
    if layout.world.goal is None:
        raise ValueError("goal must be set in layout.yaml")
    if layout.world.goal_radius is None:
        raise ValueError("goal_radius must be set in layout.yaml")
    if layout.world.start is None:
        raise ValueError("start must be set in layout.yaml")

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

    problem = ProblemSpec(
        xs=xs,
        ys=ys,
        wall_indices=wall_indices,
        obstacle_indices=obstacle_indices,
        goal_indices=goal_indices,
    )
    result = solve_laplace(
        problem,
        method=method,
        omega=omega,
        max_iter=max_iter,
        tol=tol,
    )
    trace_result = trace_path_from_start(
        problem,
        result,
        start=layout.world.start,
    )
    path_xy = trace_result.path_xy
    _ = plot_laplace(
        occ,
        xs,
        ys,
        result.phi,
        start=layout.world.start,
        goal=layout.world.goal,
        path_xy=path_xy,
    )
    _ = plot_velocity_quiver_pair(occ, xs, ys, result.u, result.v)
    _ = plot_laplace_surface_3d_pair(xs, ys, result.phi, occ=occ)
    _ = plot_residual_history(
        result.residual_history,
        result.residual_norm_history,
    )
    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
