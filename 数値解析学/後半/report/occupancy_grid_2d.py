from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict, Literal, NotRequired
import numpy as np
import numpy.typing as npt
import yaml


BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

class BoxSpec(TypedDict):
    type: Literal["box"]
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    name: NotRequired[str]


class WallSpec(TypedDict):
    type: Literal["wall"]
    x0: float
    y0: float
    x1: float
    y1: float
    name: NotRequired[str]


ObstacleSpec = BoxSpec | WallSpec


class WorldSpec(TypedDict):
    xlim: list[float]
    ylim: list[float]
    nx: int
    ny: int
    start: NotRequired[list[float]]
    goal: NotRequired[list[float]]
    goal_radius: NotRequired[float]


class LayoutSpec(TypedDict):
    world: WorldSpec
    obstacles: NotRequired[list[ObstacleSpec]]
    start: NotRequired[list[float]]
    goal: NotRequired[list[float]]
    goal_radius: NotRequired[float]
    wall_thickness: NotRequired[float]


@dataclass(frozen=True)
class World2D:
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    nx: int
    ny: int
    start: Tuple[float, float] | None = None
    goal: Tuple[float, float] | None = None
    goal_radius: float | None = None


@dataclass(frozen=True)
class Box2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    name: str = "box"


@dataclass(frozen=True)
class Wall2D:
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float
    name: str = "wall"


Obstacle2D = Box2D | Wall2D


@dataclass(frozen=True)
class Layout2D:
    world: World2D
    obstacles: List[Obstacle2D]
    wall_thickness: float | None = None


def _parse_point(
    point_raw: Any,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    name: str,
) -> Tuple[float, float]:
    if not (isinstance(point_raw, list) and len(point_raw) == 2):
        raise ValueError(f"{name} must be a list of length 2, e.g. [0.0, 1.0]")
    px, py = float(point_raw[0]), float(point_raw[1])
    if not (xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]):
        raise ValueError(f"{name} must be inside world.xlim/world.ylim")
    return px, py


def _to_world2d(
    world: WorldSpec,
    *,
    start_raw: Any | None = None,
    goal_raw: Any | None = None,
    goal_radius_raw: Any | None = None,
) -> World2D:
    xlim_raw = world["xlim"]
    ylim_raw = world["ylim"]
    nx = int(world["nx"])
    ny = int(world["ny"])

    if not (isinstance(xlim_raw, list) and len(xlim_raw) == 2):
        raise ValueError("world.xlim must be a list of length 2, e.g. [0.0, 1.0]")
    if not (isinstance(ylim_raw, list) and len(ylim_raw) == 2):
        raise ValueError("world.ylim must be a list of length 2, e.g. [0.0, 1.0]")
    if nx < 3 or ny < 3:
        raise ValueError("world.nx and world.ny must be >= 3")

    x0, x1 = float(xlim_raw[0]), float(xlim_raw[1])
    y0, y1 = float(ylim_raw[0]), float(ylim_raw[1])
    if not (x0 < x1 and y0 < y1):
        raise ValueError("world.xlim/world.ylim must satisfy min < max")

    if start_raw is None:
        start_raw = world.get("start")
    if goal_raw is None:
        goal_raw = world.get("goal")
    if goal_radius_raw is None:
        goal_radius_raw = world.get("goal_radius")
    start = None
    if start_raw is not None:
        start = _parse_point(start_raw, xlim=(x0, x1), ylim=(y0, y1), name="start")
    goal = None
    if goal_raw is not None:
        goal = _parse_point(goal_raw, xlim=(x0, x1), ylim=(y0, y1), name="goal")
    goal_radius = None
    if goal_radius_raw is not None:
        goal_radius = float(goal_radius_raw)
        if goal_radius < 0.0:
            raise ValueError("goal_radius must be >= 0")

    return World2D(
        xlim=(x0, x1),
        ylim=(y0, y1),
        nx=nx,
        ny=ny,
        start=start,
        goal=goal,
        goal_radius=goal_radius,
    )


def _to_box2d(d: BoxSpec) -> Box2D:
    name = str(d.get("name", "box"))
    xmin = float(d["xmin"])
    xmax = float(d["xmax"])
    ymin = float(d["ymin"])
    ymax = float(d["ymax"])

    _validate_box_bounds(xmin, xmax, ymin, ymax, name=name)
    return Box2D(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, name=name)


def _to_wall2d(
    d: WallSpec,
    *,
    thickness: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Wall2D:
    name = str(d.get("name", "wall"))
    x0 = float(d["x0"])
    y0 = float(d["y0"])
    x1 = float(d["x1"])
    y1 = float(d["y1"])

    if thickness <= 0.0:
        raise ValueError("wall_thickness must be > 0")
    if x0 != x1 and y0 != y1:
        raise ValueError(f"Wall '{name}' must be axis-aligned (x0==x1 or y0==y1).")

    _validate_point_in_world(x0, y0, xlim=xlim, ylim=ylim, name=f"wall '{name}'")
    _validate_point_in_world(x1, y1, xlim=xlim, ylim=ylim, name=f"wall '{name}'")
    return Wall2D(x0=x0, y0=y0, x1=x1, y1=y1, thickness=thickness, name=name)


def _validate_point_in_world(
    x: float,
    y: float,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    name: str,
) -> None:
    if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
        raise ValueError(f"{name} must be inside world.xlim/world.ylim")


def _validate_box_bounds(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    name: str,
) -> None:
    if not (xmin < xmax and ymin < ymax):
        raise ValueError(f"Invalid box '{name}': require xmin<xmax and ymin<ymax.")


def _validate_box_inside_world(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    name: str,
) -> None:
    if not (xlim[0] <= xmin and xmax <= xlim[1] and ylim[0] <= ymin and ymax <= ylim[1]):
        raise ValueError(f"Obstacle '{name}' must be inside world.xlim/world.ylim")


def _wall_to_box(wall: Wall2D) -> Box2D:
    if wall.x0 == wall.x1:
        xmin = wall.x0 - wall.thickness / 2.0
        xmax = wall.x0 + wall.thickness / 2.0
        ymin = min(wall.y0, wall.y1)
        ymax = max(wall.y0, wall.y1)
    else:
        xmin = min(wall.x0, wall.x1)
        xmax = max(wall.x0, wall.x1)
        ymin = wall.y0 - wall.thickness / 2.0
        ymax = wall.y0 + wall.thickness / 2.0
    return Box2D(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, name=wall.name)


def load_layout2d_yaml(path: str | Path) -> Layout2D:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("layout_2d.yaml root must be a mapping/dict.")
    layout = data  # type: ignore[assignment]

    world = _to_world2d(
        layout["world"],
        start_raw=layout.get("start"),
        goal_raw=layout.get("goal"),
        goal_radius_raw=layout.get("goal_radius"),
    )
    obstacles_raw = layout.get("obstacles", [])
    if not isinstance(obstacles_raw, list):
        raise ValueError("obstacles must be a list.")

    for idx, item in enumerate(obstacles_raw):
        if not isinstance(item, dict):
            raise ValueError(f"obstacles[{idx}] must be a mapping/dict.")

    wall_thickness_raw = layout.get("wall_thickness")
    wall_thickness = None
    if wall_thickness_raw is not None:
        wall_thickness = float(wall_thickness_raw)
        if wall_thickness <= 0.0:
            raise ValueError("wall_thickness must be > 0")

    obstacles: list[Obstacle2D] = []
    for o in obstacles_raw:
        otype = o.get("type", "box")
        if otype == "box":
            box = _to_box2d(o)  # type: ignore[arg-type]
            _validate_box_inside_world(
                box.xmin,
                box.xmax,
                box.ymin,
                box.ymax,
                xlim=world.xlim,
                ylim=world.ylim,
                name=box.name,
            )
            obstacles.append(box)
        elif otype == "wall":
            if wall_thickness is None:
                raise ValueError("wall_thickness is required when using wall obstacles.")
            wall = _to_wall2d(
                o,  # type: ignore[arg-type]
                thickness=wall_thickness,
                xlim=world.xlim,
                ylim=world.ylim,
            )
            wall_box = _wall_to_box(wall)
            _validate_box_inside_world(
                wall_box.xmin,
                wall_box.xmax,
                wall_box.ymin,
                wall_box.ymax,
                xlim=world.xlim,
                ylim=world.ylim,
                name=wall.name,
            )
            obstacles.append(wall)
        else:
            raise ValueError(f"Unsupported obstacle type: {otype}.")
    return Layout2D(world=world, obstacles=obstacles, wall_thickness=wall_thickness)


def rasterize_occupancy_grid_2d(
    layout: Layout2D,
) -> tuple[BoolArray, FloatArray, FloatArray]:
    """
    Returns:
        occupancy: (nx, ny) bool, True means obstacle cell
        xs: (nx,) cell-center x coordinates
        ys: (ny,) cell-center y coordinates
    """
    w = layout.world
    x0, x1 = w.xlim
    y0, y1 = w.ylim

    # cell centers (including boundary cells)
    xs = np.linspace(x0, x1, w.nx, dtype=np.float64)
    ys = np.linspace(y0, y1, w.ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (nx, ny)

    occ = np.zeros((w.nx, w.ny), dtype=np.bool_)
    for obs in layout.obstacles:
        box = _wall_to_box(obs) if isinstance(obs, Wall2D) else obs
        mask = (box.xmin <= xx) & (xx <= box.xmax) & (box.ymin <= yy) & (yy <= box.ymax)
        occ |= mask

    return occ, xs, ys


def plot_occupancy_grid(
    occ: BoolArray,
    xs: FloatArray,
    ys: FloatArray,
    *,
    start: Tuple[float, float] | None = None,
    goal: Tuple[float, float] | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    extent = (xs[0], xs[-1], ys[0], ys[-1])
    x_edges = None
    y_edges = None
    ax.imshow(
        occ.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
    )
    if xs.size > 1 and ys.size > 1:
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        x_edges = np.linspace(xs[0] - dx / 2, xs[-1] + dx / 2, xs.size + 1)
        y_edges = np.linspace(ys[0] - dy / 2, ys[-1] + dy / 2, ys.size + 1)
        # Re-render with edge-based extent so pixels align to grid.
        extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
        ax.images[-1].set_extent(extent)
        # Draw cell grid using minor ticks.
        ax.set_xticks(x_edges, minor=True)
        ax.set_yticks(y_edges, minor=True)
        ax.grid(which="minor", color="black", linewidth=0.2, alpha=0.3)
        ax.tick_params(which="minor", length=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Occupancy grid")
    ax.set_aspect("equal")
    if start is not None:
        ax.plot(start[0], start[1], marker="o", markersize=7, color="tab:blue")
    if goal is not None:
        ax.plot(goal[0], goal[1], marker="*", markersize=12, color="tab:red")
    return ax


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate and plot a 2D occupancy grid.")
    parser.add_argument(
        "--layout",
        default="",
        help="Path to layout YAML. Defaults to layout_2d.yaml in this script directory.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.layout:
        layout_path = Path(args.layout)
    else:
        layout_path = Path(__file__).resolve().parent / "layout_2d.yaml"
    layout = load_layout2d_yaml(layout_path)
    occ, xs, ys = rasterize_occupancy_grid_2d(layout)

    _ = plot_occupancy_grid(
        occ,
        xs,
        ys,
        start=layout.world.start,
        goal=layout.world.goal,
    )
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
