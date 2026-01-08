from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys

# ローカルimportを通すため、report直下を sys.path に追加する
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

from core.laplace_path_planning_solver import (
    SolveMethod,
    solve_laplace,
    trace_path_from_start,
)
from experiments.utils import build_problem_with_grid, default_layout_path
from viz.plot_laplace import (
    plot_laplace,
    plot_laplace_surface_3d_pair,
    plot_velocity_quiver_pair,
    plot_residual_history,
)


def main() -> int:
    # 入力レイアウトとソルバ設定
    layout_path = default_layout_path()
    omega = 1.5
    method = SolveMethod.SOR

    # レイアウト読み込みと必須パラメータの検証
    layout, problem, occ, xs, ys = build_problem_with_grid(layout_path)
    if layout.world.start is None:
        raise ValueError("start must be set in layout.yaml")
    # ラプラス方程式を SOR で解く
    result = solve_laplace(
        problem,
        method=method,
        omega=omega,
    )
    # 速度場に沿った経路生成
    path_result = trace_path_from_start(
        problem,
        result,
        start=layout.world.start,
    )
    path_xy = path_result.path_xy
    # 可視化（ポテンシャル/速度場/3D/残差履歴）
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
