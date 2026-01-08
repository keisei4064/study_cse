from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path
import sys

# ローカルimportを通すため、report直下を sys.path に追加する
REPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPORT_ROOT))

from core.laplace_path_planning_solver import SolveMethod, solve_laplace
from experiments.utils import (
    build_problem,
    default_layout_path,
    make_output_dir,
    plot_iterations_vs_omega,
    plot_residual_histories,
)


def _default_omegas() -> list[float]:
    return [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]


def main() -> int:
    # 入力レイアウトと掃引設定
    layout_path = default_layout_path()
    output_dir = make_output_dir("exp_omega_sweep")
    omegas = _default_omegas()

    problem = build_problem(layout_path)
    results = []
    for omega in omegas:
        result = solve_laplace(
            problem,
            method=SolveMethod.SOR,
            omega=omega,
        )
        results.append(result)

    plot_residual_histories(omegas, results, output_dir=output_dir)
    plot_iterations_vs_omega(omegas, results, output_dir=output_dir)

    # 表示
    import matplotlib.pyplot as plt
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
