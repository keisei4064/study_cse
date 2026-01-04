from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import math

from pathlib import Path
import matplotlib.pyplot as plt


# ===== Problem setting =====


def f(x: float, y: float) -> float:
    """dy/dx = f(x, y) for: x y' + 2y - x^2 = 0  =>  y' = x - 2y/x."""
    if x == 0.0:
        raise ZeroDivisionError("x=0 is not allowed for this ODE.")
    return -(2 * y) / x + x


def y_exact(x: float) -> float:
    """Analytic solution for y(1)=1: y = x^2/4 + 3/(4x^2)."""
    if x == 0.0:
        raise ZeroDivisionError("x=0 is not allowed for the exact solution.")
    return (x * x) / 4.0 + 3.0 / (4.0 * x * x)


# ===== Heun method (YOU fill the learning part) =====


def heun_step(
    f: Callable[[float, float], float],
    x_i: float,
    y_i: float,
    h: float,
) -> float:
    """
    One step of Heun's method (explicit trapezoidal / improved Euler).

    学習ポイント：
      - 予測子（Euler）: y_tilde = y_i + h * f(x_i, y_i)
      - 修正子（台形則）: y_{i+1} = y_i + (h/2) * ( f(x_i, y_i) + f(x_{i+1}, y_tilde) )
    """

    y_tilde = y_i + h * f(x_i, y_i)
    y_i_plus_1 = y_i + h / 2.0 * (f(x_i, y_i) + f(x_i + h, y_tilde))
    return y_i_plus_1


@dataclass(frozen=True)
class SolveResult:
    xs: List[float]
    ys: List[float]
    y_exact: List[float]
    err: List[float]


def solve_ivp_heun(
    x0: float,
    y0: float,
    x_end: float,
    h: float,
    *,
    f: Callable[[float, float], float] = f,
    exact: Callable[[float], float] = y_exact,
) -> SolveResult:
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if x_end <= x0:
        raise ValueError("x_end must be > x0.")

    # Step count: require exact division for clean comparison table
    span: float = x_end - x0
    n_float: float = span / h
    n: int = int(round(n_float))
    if not math.isclose(n_float, float(n), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"(x_end - x0)/h must be an integer for this template. "
            f"Got (x_end-x0)/h = {n_float}."
        )

    xs: List[float] = [x0]
    ys: List[float] = [y0]

    # 更新ループ
    for _ in range(n):
        x_i: float = xs[-1]
        y_i: float = ys[-1]
        y_next: float = heun_step(f, x_i, y_i, h)
        xs.append(x_i + h)
        ys.append(y_next)

    ys_ex: List[float] = [exact(x) for x in xs]
    err: List[float] = [abs(y_num - y_ex) for y_num, y_ex in zip(ys, ys_ex)]
    return SolveResult(xs=xs, ys=ys, y_exact=ys_ex, err=err)


def print_table(res: SolveResult, *, max_rows: int = 15) -> None:
    print(" i |    x_i    |    y_i (Heun)     |    y_i (exact)    |  abs error")
    print("---+----------+-------------------+------------------+-----------")
    for i, (x, y, ye, e) in enumerate(zip(res.xs, res.ys, res.y_exact, res.err)):
        if i >= max_rows and i != len(res.xs) - 1:
            if i == max_rows:
                print("... (omitted) ...")
            continue
        print(f"{i:2d} | {x:8.4f} | {y:17.10f} | {ye:16.10f} | {e:9.3e}")
    print()
    print(
        f"Final: x={res.xs[-1]:.4f}, y(Heun)={res.ys[-1]:.10f}, "
        f"y(exact)={res.y_exact[-1]:.10f}, abs err={res.err[-1]:.3e}"
    )


def format_h_for_name(h: float) -> str:
    return f"{h:g}".replace(".", "p")


def save_table(res: SolveResult, *, h: float) -> Path:
    tag = format_h_for_name(h)
    out_path = Path(__file__).with_name(f"report_prob2_heun_method_table_h{tag}.tsv")
    lines: List[str] = []
    lines.append("i\tx_i\ty_i (Heun)\ty_i (exact)\tabs error")
    for i, (x, y, ye, e) in enumerate(zip(res.xs, res.ys, res.y_exact, res.err)):
        lines.append(f"{i}\t{x:.3f}\t{y:.3f}\t{ye:.3f}\t{e:.3f}")
    out_path.write_text("\n".join(lines))
    return out_path


def plot_solution_h1(res_h1: SolveResult, h1: float) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res_h1.xs, res_h1.ys, marker='o', label=f'Heun h={h1:g}')
    ax.plot(res_h1.xs, res_h1.y_exact, linestyle='--', label='Exact')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Heun vs Exact')
    ax.grid(True)
    ax.legend()
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(1.0, 10.0)
    ax.set_xticks(list(range(1, 11)))

    fig.tight_layout()
    out_path = Path(__file__).with_name('report_prob2_heun_method_solution_h1.png')
    fig.savefig(out_path, dpi=150)
    plt.show()
    return out_path


def plot_error_compare(res_h1: SolveResult, h1: float, res_h2: SolveResult, h2: float) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(res_h1.xs, res_h1.err, marker='o', label=f'abs error h={h1:g}')
    ax.plot(res_h2.xs, res_h2.err, marker='o', label=f'abs error h={h2:g}')
    ax.set_xlabel('x')
    ax.set_ylabel('abs error')
    ax.set_title('Absolute Error')
    ax.grid(True)
    ax.legend()
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(1.0, 10.0)
    ax.set_xticks(list(range(1, 11)))

    fig.tight_layout()
    out_path = Path(__file__).with_name('report_prob2_heun_method_error_compare.png')
    fig.savefig(out_path, dpi=150)
    plt.show()
    return out_path


def check_second_order(x0: float, y0: float, x_end: float, h: float) -> None:
    res_h = solve_ivp_heun(x0=x0, y0=y0, x_end=x_end, h=h)
    res_h2 = solve_ivp_heun(x0=x0, y0=y0, x_end=x_end, h=h / 2.0)
    max_err_h = max(res_h.err)
    max_err_h2 = max(res_h2.err)
    ratio = max_err_h / max_err_h2
    print(
        f"2nd order check: h={h:.6g}, h/2={h/2:.6g}, "
        f"max|e(h)|={max_err_h:.3e}, max|e(h/2)|={max_err_h2:.3e}, "
        f"ratio={ratio:.3f}"
    )


def save_x_end_summary(
    res_h1: SolveResult,
    h1: float,
    res_h2: SolveResult,
    h2: float,
    x_end: float,
) -> Path:
    out_path = Path(__file__).with_name('report_prob2_heun_method_xend_summary.tsv')
    y_exact = res_h1.y_exact[-1]
    lines = [
        " \tvalue at x=10\terror",
        f"Exact\t{y_exact:.3f}\t0.000",
        f"Heun(h=1.0)\t{res_h1.ys[-1]:.3f}\t{res_h1.err[-1]:.3f}",
        f"Heun(h=0.5)\t{res_h2.ys[-1]:.3f}\t{res_h2.err[-1]:.3f}",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


def main() -> None:
    h1: float = 1.0  # ステップ幅
    h2: float = 0.5  # ステップ幅

    x0: float = 1.0
    y0: float = 1.0
    x_end: float = 10.0

    res_h1: SolveResult = solve_ivp_heun(x0=x0, y0=y0, x_end=x_end, h=h1)
    res_h2: SolveResult = solve_ivp_heun(x0=x0, y0=y0, x_end=x_end, h=h2)
    print("=== h=1.0 ===")
    print_table(res_h1, max_rows=12)
    print("=== h=0.5 ===")
    print_table(res_h2, max_rows=12)
    table_path_h1 = save_table(res_h1, h=h1)
    print(f"Saved table to {table_path_h1}")
    table_path_h2 = save_table(res_h2, h=h2)
    print(f"Saved table to {table_path_h2}")
    sol_path = plot_solution_h1(res_h1, h1)
    print(f"Saved plot to {sol_path}")
    err_path = plot_error_compare(res_h1, h1, res_h2, h2)
    print(f"Saved plot to {err_path}")
    check_second_order(x0=x0, y0=y0, x_end=x_end, h=h1)
    summary_path = save_x_end_summary(res_h1, h1, res_h2, h2, x_end)
    print(f"Saved x={x_end:g} summary to {summary_path}")


if __name__ == "__main__":
    main()
