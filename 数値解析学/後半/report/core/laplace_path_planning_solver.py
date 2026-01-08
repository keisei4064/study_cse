from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import warnings

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


DIRICHLET_WALL_VALUE = 1.0  # 外壁・障害物のディリクレ境界値
DIRICHLET_GOAL_VALUE = 0.0  # ゴール領域のディリクレ境界値
MAX_ITER = 10_000  # 最大反復回数
TOL = 1.0e-5  # 収束判定の閾値
PATH_STEP_SCALE = 0.5  # 経路生成時の1ステップに進む距離係数
PATH_MAX_STEPS = 1000  # 経路生成時の最大ステップ数


# 用いる数値解法のオプション
class SolveMethod(Enum):
    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"
    SOR = "sor"


# 問題設定をまとめたデータクラス
@dataclass(frozen=True)
class ProblemSpec:
    xs: FloatArray  # x方向の格子点座標
    ys: FloatArray  # y方向の格子点座標
    wall_indices: set[Tuple[int, int]]  # 壁のインデックス集合
    obstacle_indices: set[Tuple[int, int]]  # 障害物のインデックス集合
    goal_indices: set[Tuple[int, int]]  # ゴール領域のインデックス集合


# ラプラス方程式の計算結果をまとめたデータクラス
@dataclass(frozen=True)
class LaplaceResult:
    phi: FloatArray  # ポテンシャル場
    u: FloatArray  # x方向の勾配
    v: FloatArray  # y方向の勾配
    iterations: int  # 反復回数
    residual_history: list[float]  # 残差の履歴
    residual_norm_history: list[float]  # 正規化した残差の履歴


# 経路生成の結果をまとめたデータクラス
@dataclass(frozen=True)
class PathResult:
    path_xy: list[Tuple[float, float]]  # 生成された経路


def solve_laplace(
    problem: ProblemSpec,
    *,
    method: SolveMethod = SolveMethod.SOR,
    omega: float | None = 1.5,
) -> LaplaceResult:
    """
    2D ラプラス方程式を反復法で解く

    入力:
        problem: 問題設定
        method: 反復法の種類
        omega: SOR の緩和係数（1 <= omega < 2）

    返り値:
        LaplaceResult: 計算結果

    境界条件:
        - 外壁/障害物は Dirichlet: phi = DIRICHLET_WALL_VALUE
        - ゴール領域は Dirichlet: phi = DIRICHLET_GOAL_VALUE
    """

    if method == SolveMethod.SOR:
        assert omega is not None, "SOR では omega の指定が必要"
        assert 1.0 <= omega < 2.0, "omega は必ず 1 以上 2 未満"

    # 定数値計算
    xs = problem.xs
    ys = problem.ys
    nx, ny = xs.size, ys.size
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    dx2 = dx * dx
    dy2 = dy * dy
    L = 2.0 / dx2 + 2.0 / dy2

    # Dirichlet条件に該当するインデックスを設定（外壁 + 障害物 + ゴール）
    is_dirichlet = np.zeros((nx, ny), dtype=np.bool_)
    for i, j in problem.wall_indices:
        is_dirichlet[i, j] = True
    for i, j in problem.obstacle_indices:
        is_dirichlet[i, j] = True
    for i, j in problem.goal_indices:
        is_dirichlet[i, j] = True

    # ポテンシャル場の初期値設定
    phi = np.full((nx, ny), DIRICHLET_WALL_VALUE, dtype=np.float64)
    phi[is_dirichlet] = DIRICHLET_WALL_VALUE  # 壁
    for i, j in problem.goal_indices:
        phi[i, j] = DIRICHLET_GOAL_VALUE  # ゴール

    # 反復計算の記録変数を用意
    res_max0: float | None = None
    iterations = 0
    residual_history: list[float] = []
    residual_norm_history: list[float] = []

    # ポテンシャルを計算
    for iter_count in range(1, MAX_ITER + 1):
        iterations = iter_count
        res_max = 0.0

        if iter_count % 2 == 1:
            # 奇数回
            i_range = range(1, nx - 1)
            j_range = range(1, ny - 1)
        else:
            # 偶数回
            i_range = range(nx - 2, 0, -1)
            j_range = range(ny - 2, 0, -1)

        # 使用する手法ごとに場合分け
        match method:
            # ヤコビ法
            case SolveMethod.JACOBI:
                phi_prev = phi.copy()
                for i in i_range:
                    for j in j_range:
                        if is_dirichlet[i, j]:
                            continue

                        # 残差 r の計算（前回反復の値を使う）
                        r = (
                            phi_prev[i - 1, j] - 2 * phi_prev[i, j] + phi_prev[i + 1, j]
                        ) / dx2 + (
                            phi_prev[i, j - 1] - 2 * phi_prev[i, j] + phi_prev[i, j + 1]
                        ) / dy2  # 2Dラプラシアン

                        # ヤコビ法による更新
                        phi[i, j] = phi_prev[i, j] + r / L

                        if abs(r) > res_max:
                            res_max = abs(r)

            # ガウス-ザイデル法
            case SolveMethod.GAUSS_SEIDEL:
                for i in i_range:
                    for j in j_range:
                        if is_dirichlet[i, j]:
                            continue

                        # 残差 r の計算
                        r = (phi[i - 1, j] - 2 * phi[i, j] + phi[i + 1, j]) / dx2 + (
                            phi[i, j - 1] - 2 * phi[i, j] + phi[i, j + 1]
                        ) / dy2  # 2Dラプラシアン

                        # ガウスザイデル法による更新
                        phi[i, j] = phi[i, j] + r / L

                        if abs(r) > res_max:
                            res_max = abs(r)

            # SOR法
            case SolveMethod.SOR:
                for i in i_range:
                    for j in j_range:
                        if is_dirichlet[i, j]:
                            continue

                        # 残差 r の計算
                        r = (phi[i - 1, j] - 2 * phi[i, j] + phi[i + 1, j]) / dx2 + (
                            phi[i, j - 1] - 2 * phi[i, j] + phi[i, j + 1]
                        ) / dy2  # 2Dラプラシアン

                        # SOR法による更新
                        phi[i, j] = phi[i, j] + omega * r / L

                        if abs(r) > res_max:
                            res_max = abs(r)

        # 正規化した残差を計算
        if res_max0 is None:
            res_max0 = res_max
        assert res_max0, "絶対に数値持ってるはず"
        if res_max0 == 0:
            # （初回で解なら終了）
            break
        res_norm = res_max / res_max0

        # 履歴に記録
        residual_history.append(res_max)
        residual_norm_history.append(res_norm)

        # 終了判定
        if res_norm <= TOL:
            break

    # 勾配を計算
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    for i in range(nx):
        for j in range(ny):
            # 通常は両側差分近似，端だけ片側差分近似になる
            im1 = max(i - 1, 0)
            ip1 = min(i + 1, nx - 1)
            jm1 = max(j - 1, 0)
            jp1 = min(j + 1, ny - 1)

            # 格子間距離
            dx1 = xs[ip1] - xs[im1]
            dy1 = ys[jp1] - ys[jm1]

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
) -> PathResult:
    """
    速度場に沿って経路を生成する

    入力:
        problem: 問題設定（格子・ゴール領域）
        trace_input: ラプラス方程式の計算結果
        start: 始点座標

    返り値:
        PathResult:　生成された経路

    生成停止条件:
        - 計算領域の外へ出た
        - ゴール領域のセルに到達した
        - 勾配がゼロで進行不能になった
        - 最大ステップ数に到達した
    """

    # 定数計算
    xs = problem.xs
    ys = problem.ys
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    step = PATH_STEP_SCALE * min(dx, dy)
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])

    # 軌道生成
    path: list[Tuple[float, float]] = [start]
    x, y = start
    for _ in range(PATH_MAX_STEPS):
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            # 領域外に出たため終了
            break

        # 現在位置に最も近い格子点を選ぶ
        i = int(np.argmin(np.abs(xs - x)))
        j = int(np.argmin(np.abs(ys - y)))
        if (i, j) in problem.goal_indices:
            # ゴール領域に到達したため終了
            break

        # 低ポテンシャル側へ進む
        ux = -float(trace_input.u[i, j])
        vy = -float(trace_input.v[i, j])
        norm = (ux * ux + vy * vy) ** 0.5
        if norm == 0.0:
            # 勾配が存在しないため終了
            warnings.warn(
                "勾配がゼロのため経路生成を終了します",
                RuntimeWarning,
            )
            break
        x += step * ux / norm
        y += step * vy / norm

        # ステップ保存
        path.append((x, y))

    return PathResult(path_xy=path)
