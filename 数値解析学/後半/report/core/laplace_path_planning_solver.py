from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


DIRICHLET_WALL_VALUE = 1.0  # 外壁・障害物のディリクレ境界値
DIRICHLET_GOAL_VALUE = 0.0  # ゴール領域のディリクレ境界値
TRACE_STEP_SCALE = 0.5  # 経路追跡の1ステップ長さ係数（格子幅に対する比）
TRACE_MAX_STEPS = 5000  # 経路追跡の最大ステップ数


class SolveMethod(Enum):
    SOR = "sor"


@dataclass(frozen=True)
class ProblemSpec:
    xs: FloatArray
    ys: FloatArray
    wall_indices: set[Tuple[int, int]]
    obstacle_indices: set[Tuple[int, int]]
    goal_indices: set[Tuple[int, int]]


@dataclass(frozen=True)
class LaplaceResult:
    phi: FloatArray
    u: FloatArray
    v: FloatArray
    iterations: int
    residual_history: list[float]
    residual_norm_history: list[float]


@dataclass(frozen=True)
class TraceResult:
    path_xy: list[Tuple[float, float]]
    steps: int


def solve_laplace(
    problem: ProblemSpec,
    *,
    method: SolveMethod = SolveMethod.SOR,
    omega: float | None = 1.5,
    max_iter: int = 10_000,
    tol: float = 1.0e-5,
) -> LaplaceResult:
    """
    2D Laplace 方程式を反復法で解く（数値計算部分のみ）。

    境界条件:
        - 外壁は Dirichlet: u = boundary_value (>0)
        - 障害物内部は Dirichlet（計算スキップ）
        - ゴール領域は Dirichlet: u = DIRICHLET_GOAL_VALUE
        - スタート点は固定しない
    """
    xs = problem.xs
    ys = problem.ys
    nx, ny = xs.size, ys.size
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dx2 = dx * dx
    dy2 = dy * dy
    L = 2.0 / dx2 + 2.0 / dy2

    # ポテンシャル場の初期値
    phi = np.full((nx, ny), DIRICHLET_WALL_VALUE, dtype=np.float64)

    # Dirichlet で固定するセル（外壁 + 障害物 + ゴール）
    is_dirichlet = np.zeros((nx, ny), dtype=np.bool_)
    for i, j in problem.wall_indices:
        is_dirichlet[i, j] = True
    for i, j in problem.obstacle_indices:
        is_dirichlet[i, j] = True
    for i, j in problem.goal_indices:
        is_dirichlet[i, j] = True

    # ---- 初期条件の設定 ----
    # 外壁/障害物/ゴールをそれぞれ指定値で設定
    phi[is_dirichlet] = DIRICHLET_WALL_VALUE
    for i, j in problem.goal_indices:
        phi[i, j] = DIRICHLET_GOAL_VALUE

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

    return LaplaceResult(
        phi=phi,
        u=u,
        v=v,
        iterations=iterations,
        residual_history=residual_history,
        residual_norm_history=residual_norm_history,
    )


def trace_path_from_start(
    problem: ProblemSpec,
    trace_input: LaplaceResult,
    *,
    start: Tuple[float, float],
) -> TraceResult:
    xs = problem.xs
    ys = problem.ys
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    step = TRACE_STEP_SCALE * min(dx, dy)
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])

    path: list[Tuple[float, float]] = [start]
    x, y = start
    steps = 0
    for _ in range(TRACE_MAX_STEPS):
        steps += 1
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            break

        # 現在位置に最も近い格子点を選ぶ
        i = int(np.argmin(np.abs(xs - x)))
        j = int(np.argmin(np.abs(ys - y)))
        if (i, j) in problem.goal_indices:
            break

        # 低ポテンシャル側へ進む
        ux = -float(trace_input.u[i, j])
        vy = -float(trace_input.v[i, j])
        norm = (ux * ux + vy * vy) ** 0.5
        if norm == 0.0:
            break
        x += step * ux / norm
        y += step * vy / norm
        path.append((x, y))

    return TraceResult(path_xy=path, steps=steps)
