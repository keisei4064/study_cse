from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import numpy.typing as npt


BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


def _path_to_xy(
    path: Iterable[Tuple[int, int]],
    xs: FloatArray,
    ys: FloatArray,
) -> Tuple[FloatArray, FloatArray]:
    indices = list(path)
    ix = np.array([p[0] for p in indices], dtype=int)
    iy = np.array([p[1] for p in indices], dtype=int)
    return xs[ix], ys[iy]


def plot_laplace_2d(
    occ: BoolArray,
    xs: FloatArray,
    ys: FloatArray,
    phi: FloatArray,
    *,
    start: Tuple[float, float] | None = None,
    goal: Tuple[float, float] | None = None,
    path: Iterable[Tuple[int, int]] | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    extent = (xs[0], xs[-1], ys[0], ys[-1])
    ax.imshow(
        occ.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
    )
    ax.contour(xs, ys, phi.T, levels=20, cmap="viridis", alpha=0.7)

    if path is not None:
        path_x, path_y = _path_to_xy(path, xs, ys)
        ax.plot(path_x, path_y, color="tab:orange", linewidth=2.0)
    if start is not None:
        ax.plot(start[0], start[1], marker="o", markersize=7, color="tab:blue")
    if goal is not None:
        ax.plot(goal[0], goal[1], marker="*", markersize=12, color="tab:red")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Laplace solution (2D)")
    ax.set_aspect("equal")
    return ax


def plot_residual_history(
    residual: Iterable[float],
    residual_norm: Iterable[float],
    *,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(list(residual), label="residual")
    ax.plot(list(residual_norm), label="residual (normalized)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.set_yscale("log")
    ax.set_title("Residual history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_laplace_surface_3d(
    xs: FloatArray,
    ys: FloatArray,
    phi: FloatArray,
    *,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ax.plot_surface(X, Y, phi, cmap="viridis", linewidth=0.0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi")
    ax.set_title("Laplace potential (3D surface)")
    return ax
