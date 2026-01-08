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
from experiments.utils import build_problem_with_grid, default_layout_path, make_output_dir
from viz.plot_laplace import (
    plot_laplace,
    plot_laplace_surface_3d_pair,
    plot_velocity_quiver_pair,
    plot_residual_history,
)


def _save_axis_figure(ax, path: Path) -> None:
    fig = ax.get_figure()
    assert fig is not None
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def _set_suptitle(fig, title: str) -> None:
    fig.suptitle(title)


def main() -> int:
    # 入力レイアウトとソルバ設定
    layout_path = default_layout_path()
    output_dir = make_output_dir("exp_sor_only")
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
    ax_lin, ax_log = plot_laplace(
        occ,
        xs,
        ys,
        result.phi,
        start=layout.world.start,
        goal=layout.world.goal,
        path_xy=path_xy,
    )
    fig_laplace = ax_lin.get_figure()
    assert fig_laplace is not None
    _set_suptitle(fig_laplace, "Potential Field")
    _save_axis_figure(ax_lin, output_dir / "potential_field.png")

    ax_vel_lin, ax_vel_log = plot_velocity_quiver_pair(occ, xs, ys, result.u, result.v)
    fig_vel = ax_vel_lin.get_figure()
    assert fig_vel is not None
    _set_suptitle(fig_vel, "Gradient Descent Flow")
    _save_axis_figure(ax_vel_lin, output_dir / "gradient_descent_flow.png")

    ax_surf_lin, ax_surf_log = plot_laplace_surface_3d_pair(xs, ys, result.phi, occ=occ)
    fig_surf = ax_surf_lin.get_figure()
    assert fig_surf is not None
    _set_suptitle(fig_surf, "Potential Field (3D)")
    _save_axis_figure(ax_surf_lin, output_dir / "potential_field_3d.png")

    fig_res, _ = plot_residual_history(
        result.residual_history,
        result.residual_norm_history,
    )
    fig_res.suptitle("Residual History")
    fig_res.tight_layout()
    fig_res.savefig(output_dir / "residual_history.png", dpi=200)
    import matplotlib.pyplot as plt

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
