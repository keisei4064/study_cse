from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import time
import csv

from core.laplace_path_planning_solver import ProblemSpec
from core.path_planning_utils import goal_disk_indices
from problem_gen.occupancy_grid import (
    Layout,
    World,
    load_layout_yaml,
    rasterize_occupancy_grid,
)


def ensure_report_on_syspath() -> None:
    # ローカルimportを通すため、report直下を sys.path に追加する
    report_root = Path(__file__).resolve().parents[1]
    if str(report_root) not in sys.path:
        sys.path.insert(0, str(report_root))


def default_layout_path() -> Path:
    return Path(__file__).resolve().parents[1] / "problem_gen" / "layout.yaml"


def make_output_dir(exp_name: str) -> Path:
    output_dir = Path(__file__).resolve().parent / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_problem(layout_path: Path) -> ProblemSpec:
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


def build_problem_with_layout(layout_path: Path):
    layout = load_layout_yaml(layout_path)
    problem = build_problem(layout_path)
    return layout, problem


def build_problem_with_grid(layout_path: Path):
    layout = load_layout_yaml(layout_path)
    occ, xs, ys = rasterize_occupancy_grid(layout)
    problem = build_problem(layout_path)
    return layout, problem, occ, xs, ys


def build_problem_with_grid_size(layout_path: Path, nx: int, ny: int):
    layout = load_layout_yaml(layout_path)
    world = layout.world
    if world.goal is None:
        raise ValueError("goal must be set in layout.yaml")
    if world.goal_radius is None:
        raise ValueError("goal_radius must be set in layout.yaml")
    world_resized = World(
        xlim=world.xlim,
        ylim=world.ylim,
        nx=nx,
        ny=ny,
        start=world.start,
        goal=world.goal,
        goal_radius=world.goal_radius,
    )
    layout_resized = Layout(
        world=world_resized,
        obstacles=layout.obstacles,
        wall_thickness=layout.wall_thickness,
    )
    occ, xs, ys = rasterize_occupancy_grid(layout_resized)

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
            world_resized.goal,
            world_resized.goal_radius,
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
    return layout_resized, problem, occ, xs, ys


def plot_residual_histories(
    omegas: Iterable[float],
    results,
    *,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for omega, result in zip(omegas, results, strict=True):
        ax.plot(result.residual_norm_history, label=rf"$\omega$={omega:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual (normalized)")
    ax.set_title("SOR residual histories")
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "residual_histories.png", dpi=200)


def plot_iterations_vs_omega(
    omegas: Iterable[float],
    results,
    *,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    iterations = [r.iterations for r in results]
    fig, ax = plt.subplots()
    ax.plot(list(omegas), iterations, marker="o")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("iterations to convergence")
    ax.set_xlim(1.0, 2.0)
    ax.set_title("Iterations to Convergence vs $\\omega$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "iterations_vs_omega.png", dpi=200)


def measure_cpu_time(fn, *args, **kwargs):
    """Measure CPU time for a function call."""
    t0 = time.process_time()
    result = fn(*args, **kwargs)
    dt = time.process_time() - t0
    return result, dt


def write_timing_csv(output_dir: Path, rows: list[tuple[str, float]]) -> None:
    output_path = output_dir / "timing.csv"
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "seconds"])
        for label, seconds in rows:
            writer.writerow([label, f"{seconds:.6f}"])
