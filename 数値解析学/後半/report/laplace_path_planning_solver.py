from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import numpy.typing as npt


BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


DIRICHLET_WALL_VALUE = 1.0

DirichletCondition = Tuple[Tuple[int, int], float]


class SolveMethod(Enum):
    SOR = "sor"


@dataclass(frozen=True)
class SolveResult:
    phi: FloatArray
    u: FloatArray
    v: FloatArray
    iterations: int
    residual_history: list[float]
    residual_norm_history: list[float]


def solve_laplace(
    occ: BoolArray,
    xs: FloatArray,
    ys: FloatArray,
    *,
    dirichlet_conditions: list[DirichletCondition],
    method: SolveMethod = SolveMethod.SOR,
    omega: float | None = 1.5,
    max_iter: int = 10_000,
    tol: float = 1.0e-5,
    boundary_value: float = DIRICHLET_WALL_VALUE,
) -> SolveResult:
    """
    2D Laplace 方程式を反復法で解く（数値計算部分のみ）。

    境界条件:
        - 外壁 + 障害物は Dirichlet: u = boundary_value (>0)
        - dirichlet_conditions で指定した点は Dirichlet（個別値）
        - スタート点は固定しない
    """
    nx, ny = occ.shape
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dx2 = dx * dx
    dy2 = dy * dy
    L = 2.0 / dx2 + 2.0 / dy2

    # ポテンシャル場の初期値
    phi = np.full((nx, ny), boundary_value, dtype=np.float64)

    # Dirichlet で固定するセル（外壁 + 障害物 + dirichlet_conditions）
    is_dirichlet = np.zeros_like(occ, dtype=np.bool_)
    is_dirichlet[0, :] = True
    is_dirichlet[-1, :] = True
    is_dirichlet[:, 0] = True
    is_dirichlet[:, -1] = True
    is_dirichlet |= occ
    for (i, j), _ in dirichlet_conditions:
        is_dirichlet[i, j] = True

    # ---- 初期条件の設定 ----
    # 外壁/障害物/dirichlet_conditions をそれぞれ指定値で設定
    phi[is_dirichlet] = boundary_value
    phi[0, :] = boundary_value
    phi[-1, :] = boundary_value
    phi[:, 0] = boundary_value
    phi[:, -1] = boundary_value
    for (i, j), value in dirichlet_conditions:
        phi[i, j] = value

    res_max0: float | None = None
    iterations = 0
    residual_history: list[float] = []
    residual_norm_history: list[float] = []

    for iter_count in range(1, max_iter + 1):
        iterations = iter_count
        res_max = 0.0

        if iter_count % 2 == 1:  # 奇数回
            i_range = range(1, nx - 1)
            j_range = range(1, ny - 1)
        else:  # 偶数回
            i_range = range(nx - 2, 0, -1)
            j_range = range(ny - 2, 0, -1)

        match method:
            case SolveMethod.SOR:
                if omega is None:
                    raise ValueError("omega is required for SOR")
                for i in i_range:
                    for j in j_range:
                        if is_dirichlet[i, j]:
                            continue

                        # 残差 r の計算
                        r = (phi[i - 1, j] - 2 * phi[i, j] + phi[i + 1, j]) / dx2 + (
                            phi[i, j - 1] - 2 * phi[i, j] + phi[i, j + 1]
                        ) / dy2  # 2Dラプラシアン

                        # SOR 更新
                        phi[i, j] += omega * r / L

                        if abs(r) > res_max:
                            res_max = abs(r)
            case _:
                raise ValueError(f"Unsupported method: {method}")

        if res_max0 is None:
            res_max0 = res_max
        residual_history.append(res_max)
        assert res_max0 is not None  # 絶対数値持ってるはず
        res_norm = res_max / res_max0 if res_max0 != 0.0 else 0.0
        residual_norm_history.append(res_norm)
        if res_max0 == 0.0:
            break
        if res_norm <= tol:
            break

    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    for i in range(nx):
        for j in range(ny):
            im1 = max(i - 1, 0)
            ip1 = min(i + 1, nx - 1)
            jm1 = max(j - 1, 0)
            jp1 = min(j + 1, ny - 1)

            dx1 = xs[ip1] - xs[im1]
            dy1 = ys[jp1] - ys[jm1]
            if dx1 == 0.0 or dy1 == 0.0:
                continue

            u[i, j] = (phi[ip1, j] - phi[im1, j]) / dx1
            v[i, j] = (phi[i, jp1] - phi[i, jm1]) / dy1

    return SolveResult(
        phi=phi,
        u=u,
        v=v,
        iterations=iterations,
        residual_history=residual_history,
        residual_norm_history=residual_norm_history,
    )


def _closest_index(values: FloatArray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def goal_disk_indices(
    goal: Tuple[float, float],
    radius: float,
    xs: FloatArray,
    ys: FloatArray,
) -> list[Tuple[int, int]]:
    nx, ny = xs.size, ys.size
    gx, gy = goal
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    r_ix = int(np.ceil(radius / dx))
    r_iy = int(np.ceil(radius / dy))

    ig = _closest_index(xs, gx)
    jg = _closest_index(ys, gy)
    indices: list[Tuple[int, int]] = []
    for i in range(max(0, ig - r_ix), min(nx, ig + r_ix + 1)):
        for j in range(max(0, jg - r_iy), min(ny, jg + r_iy + 1)):
            if (xs[i] - gx) ** 2 + (ys[j] - gy) ** 2 <= radius * radius:
                indices.append((i, j))
    return indices


def trace_path_from_start(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    goal_radius: float,
    xs: FloatArray,
    ys: FloatArray,
    u: FloatArray,
    v: FloatArray,
    *,
    max_steps: int = 5000,
    step_scale: float = 0.5,
) -> list[Tuple[float, float]]:
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    step = step_scale * min(dx, dy)
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])

    path: list[Tuple[float, float]] = [start]
    x, y = start
    # TODO(report): 経路生成が連続座標ステップなので、レポートで妥当性/比較を要検討。
    for _ in range(max_steps):
        if (x - goal[0]) ** 2 + (y - goal[1]) ** 2 <= goal_radius * goal_radius:
            break
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            break

        i = _closest_index(xs, x)
        j = _closest_index(ys, y)

        # 低ポテンシャル側へ進む
        ux = -float(u[i, j])
        vy = -float(v[i, j])
        norm = (ux * ux + vy * vy) ** 0.5
        if norm == 0.0:
            break
        x += step * ux / norm
        y += step * vy / norm
        path.append((x, y))

    return path
