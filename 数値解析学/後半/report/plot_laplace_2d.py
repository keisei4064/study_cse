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
    path_xy: Iterable[Tuple[float, float]] | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    else:
        ax_lin, ax_log = ax

    extent = (xs[0], xs[-1], ys[0], ys[-1])
    ax_lin.imshow(
        occ.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
    )
    boundary_mask = np.zeros_like(occ, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    phi_masked = np.ma.masked_array(phi, mask=(occ | boundary_mask))
    im_lin = ax_lin.imshow(
        phi_masked.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )

    ax_log.imshow(
        occ.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
    )
    phi_inv = 1.0 - phi_masked
    phi_clipped = np.clip(phi_inv, 1.0e-12, None)
    phi_log = -np.log10(phi_clipped)
    im_log = ax_log.imshow(
        phi_log.T,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )

    if path is not None:
        path_x, path_y = _path_to_xy(path, xs, ys)
        ax_lin.plot(path_x, path_y, color="tab:orange", linewidth=2.0)
        ax_log.plot(path_x, path_y, color="tab:orange", linewidth=2.0)
    if path_xy is not None:
        points = list(path_xy)
        if points:
            px = [p[0] for p in points]
            py = [p[1] for p in points]
            ax_lin.plot(px, py, color="tab:orange", linewidth=2.0)
            ax_log.plot(px, py, color="tab:orange", linewidth=2.0)
    if start is not None:
        ax_lin.plot(start[0], start[1], marker="o", markersize=7, color="tab:blue")
        ax_log.plot(start[0], start[1], marker="o", markersize=7, color="tab:blue")
    if goal is not None:
        ax_lin.plot(goal[0], goal[1], marker="*", markersize=12, color="tab:red")
        ax_log.plot(goal[0], goal[1], marker="*", markersize=12, color="tab:red")

    ax_lin.set_xlabel("x")
    ax_lin.set_ylabel("y")
    ax_lin.set_title("Laplace solution (linear)")
    ax_lin.set_aspect("equal")
    plt.colorbar(im_lin, ax=ax_lin, fraction=0.046, pad=0.04)
    ax_log.set_xlabel("x")
    ax_log.set_title("Laplace solution (-log10(1 - phi))")
    ax_log.set_aspect("equal")
    plt.colorbar(im_log, ax=ax_log, fraction=0.046, pad=0.04)
    return (ax_lin, ax_log)


def plot_velocity_quiver_2d(
    occ: BoolArray,
    xs: FloatArray,
    ys: FloatArray,
    u: FloatArray,
    v: FloatArray,
    *,
    step: int | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    stride = step if step is not None else max(1, min(xs.size, ys.size) // 20)
    mask_occ = occ[::stride, ::stride]
    uu = u[::stride, ::stride]
    vv = v[::stride, ::stride]
    speed = np.sqrt(uu**2 + vv**2)
    speed_safe = np.where(speed == 0.0, 1.0, speed)
    uu_unit = uu / speed_safe
    vv_unit = vv / speed_safe

    grid_scale = 0.5 * min(float(xs[1] - xs[0]), float(ys[1] - ys[0]))
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    ax.imshow(
        occ.T,
        origin="lower",
        extent=extent,
        cmap="gray_r",
        interpolation="nearest",
    )
    X = xs[::stride]
    Y = ys[::stride]
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    valid = ~mask_occ
    q = ax.quiver(
        XX[valid],
        YY[valid],
        (uu_unit * grid_scale)[valid],
        (vv_unit * grid_scale)[valid],
        speed[valid],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Velocity field (quiver)")
    ax.set_aspect("equal")
    plt.colorbar(q, ax=ax, label="|v|")
    return ax


def plot_residual_history(
    residual: Iterable[float],
    residual_norm: Iterable[float],
):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    ax0.plot(list(residual), color="tab:blue")
    ax0.set_ylabel("residual")
    ax0.set_yscale("log")
    ax0.set_title("Residual history")
    ax0.grid(True, alpha=0.3)

    ax1.plot(list(residual_norm), color="tab:orange")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("residual (normalized)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    return fig, (ax0, ax1)


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
    else:
        fig = ax.figure

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    surf = ax.plot_surface(X, Y, phi, cmap="viridis", linewidth=0.0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi", labelpad=8)
    ax.set_title("Laplace potential (3D surface)")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.15)
    return ax


def plot_laplace_surface_3d_log(
    xs: FloatArray,
    ys: FloatArray,
    phi: FloatArray,
    *,
    occ: BoolArray | None = None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    phi_inv = 1.0 - phi
    if occ is not None:
        boundary_mask = np.zeros_like(occ, dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        phi_inv = np.ma.masked_array(phi_inv, mask=(occ | boundary_mask))
    phi_log = -np.log10(np.clip(phi_inv, 1.0e-12, None))
    surf = ax.plot_surface(X, Y, phi_log, cmap="viridis", linewidth=0.0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("-log10(1 - phi)", labelpad=8)
    ax.set_title("Laplace potential (3D surface, -log10(1 - phi))")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.15)
    return ax


def plot_laplace_surface_3d_pair(
    xs: FloatArray,
    ys: FloatArray,
    phi: FloatArray,
    *,
    occ: BoolArray | None = None,
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax_lin = fig.add_subplot(1, 2, 1, projection="3d")
    ax_log = fig.add_subplot(1, 2, 2, projection="3d")
    plot_laplace_surface_3d(xs, ys, phi, ax=ax_lin)
    plot_laplace_surface_3d_log(xs, ys, phi, occ=occ, ax=ax_log)
    fig.tight_layout()
    return ax_lin, ax_log
