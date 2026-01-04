from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence
import math
import numpy as np
import numpy.typing as npt
from pathlib import Path


Vec2 = npt.NDArray[np.float64]
Vec2Like = Sequence[float] | npt.NDArray[np.floating]
Rhs2 = Callable[[float, Vec2], Vec2]  # “right-hand side”（方程式の右辺）

# ===== 実装すべき中核ロジック（RK4 1ステップ） =====


def rk4_step(rhs: Rhs2, x: float, y: Vec2, h: float) -> Vec2:
    """
    2次元系 y' = rhs(x, y) に対する古典的4次のルンゲ・クッタ法の1ステップ
    """
    k1 = rhs(x, y)
    k2 = rhs(x + h / 2, y + (h / 2) * k1)
    k3 = rhs(x + h / 2, y + (h / 2) * k2)
    k4 = rhs(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ===== 初期値問題に対する積分器 =====


@dataclass(frozen=True)
class Ivpsol:
    """Initial Value Problem Solution"""

    xs: List[float]
    ys: List[Vec2]


def integrate_ivp(
    rhs: Rhs2,
    x0: float,
    y0: Vec2Like,
    x_end: float,
    h: float,
) -> Ivpsol:
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if x_end <= x0:
        raise ValueError("x_end must be > x0.")

    span: float = x_end - x0
    n_float: float = span / h
    n: int = int(round(n_float))
    if not math.isclose(n_float, float(n), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"(x_end-x0)/h must be integer. Got {n_float}.")

    xs: List[float] = [x0]
    ys: List[Vec2] = [np.asarray(y0, dtype=float)]

    for _ in range(n):
        x_i: float = xs[-1]
        y_i: Vec2 = ys[-1]
        y_next: Vec2 = rk4_step(rhs, x_i, y_i, h)
        xs.append(x_i + h)
        ys.append(y_next)

    return Ivpsol(xs=xs, ys=ys)


# ===== 問題固有の右辺 =====
# uについて: u'' - u + x = 0  -> u'' = u - x
# 状態を (u, u') = (u1, u2) とすると: u1' = u2, u2' = u1 - x


def rhs_u(x: float, s: Vec2) -> Vec2:
    u1: float = float(s[0])
    u2: float = float(s[1])
    return np.array([u2, u1 - x], dtype=float)


# zについて: z'' - z = 0 -> z'' = z
# 状態を (z, z') = (z1, z2) とすると: z1' = z2, z2' = z1


def rhs_z(x: float, s: Vec2) -> Vec2:
    z1: float = float(s[0])
    z2: float = float(s[1])
    return np.array([z2, z1], dtype=float)


# ===== 重ね合わせ =====


@dataclass(frozen=True)
class BvpResult:
    """BVP は “Boundary Value Problem”（境界値問題）"""

    xs: List[float]
    y_num: List[float]
    y_ex: List[float]
    abs_err: List[float]
    C: float
    u_end: float
    z_end: float


def y_exact(x: float) -> float:
    # 厳密解: y = x - sinh(x)/sinh(1)
    return x - math.sinh(x) / math.sinh(1.0)


def solve_bvp_superposition(
    h: float,
    *,
    uprime0: float = 0.0,  # 仮定 u'(0)
    zprime0: float = 1.0,  # 仮定 z'(0) != 0
    x0: float = 0.0,
    x_end: float = 1.0,
    y0: float = 0.0,  # y(0)=0
    y_end: float = 0.0,  # y(1)=0
) -> BvpResult:
    # if zprime0 == 0.0:
    #     raise ValueError("zprime0 must be non-zero.")

    # u(0)=y0, u'(0)=s
    sol_u: Ivpsol = integrate_ivp(
        rhs=lambda x, st: rhs_u(x, st), x0=x0, y0=(y0, uprime0), x_end=x_end, h=h
    )

    # z(0)=0, z'(0)=alpha
    sol_z: Ivpsol = integrate_ivp(
        rhs=lambda x, st: rhs_z(x, st), x0=x0, y0=(0.0, zprime0), x_end=x_end, h=h
    )

    xs: List[float] = sol_u.xs  # 同じ刻み幅の格子
    u_vals: List[float] = [st[0] for st in sol_u.ys]  # uのみ取り出す
    z_vals: List[float] = [st[0] for st in sol_z.ys]  # zのみ取り出す

    u_end_val: float = u_vals[-1]
    z_end_val: float = z_vals[-1]
    if abs(z_end_val) < 1e-14:
        raise ZeroDivisionError(
            "z(1) is too small; C would blow up (numerically unstable setting)."
        )

    # 重ね合わせ係数を求める
    # y(1) = u(1) + C z(1) = y_end  -> C = (y_end - u(1)) / z(1)
    C: float = (y_end - u_end_val) / z_end_val

    y_num: List[float] = [u + C * z for (u, z) in zip(u_vals, z_vals)]
    y_ex: List[float] = [y_exact(x) for x in xs]
    abs_err: List[float] = [abs(a - b) for (a, b) in zip(y_num, y_ex)]

    return BvpResult(
        xs=xs,
        y_num=y_num,
        y_ex=y_ex,
        abs_err=abs_err,
        C=C,
        u_end=u_end_val,
        z_end=z_end_val,
    )


# ===== 表示用ヘルパ =====


def print_summary(res: BvpResult, *, label: str) -> None:
    max_err: float = max(res.abs_err)
    end_err: float = res.abs_err[-1]
    print(f"[{label}] C={res.C:.10f}, u(1)={res.u_end:.10f}, z(1)={res.z_end:.10f}")
    print(f"[{label}] max|err|={max_err:.3e}, |err(1)|={end_err:.3e}")
    print(f"[{label}] y_num(1)={res.y_num[-1]:.10f}  (should be 0)")


def compare_assumptions(
    h: float,
    *,
    uprime0_list: Sequence[float],
    zprime_list: Sequence[float],
) -> None:
    print("== 仮定の感度 (u'(0)=uprime0, z'(0)=zprime0) ==")
    for uprime0 in uprime0_list:
        for zprime0 in zprime_list:
            res: BvpResult = solve_bvp_superposition(
                h, uprime0=uprime0, zprime0=zprime0
            )
            print_summary(
                res, label=f"h={h:g}, uprime0={uprime0:g}, zprime0={zprime0:g}"
            )
    print()


# ===== （任意）プロット =====


def plot_assumption_grids(
    h: float, *, uprime0_list: Sequence[float], zprime_list: Sequence[float]
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    out_dir = Path(__file__).resolve().parent

    uprime0_vals = np.asarray(uprime0_list, dtype=float)
    zprime_vals = np.asarray(zprime_list, dtype=float)
    max_err_grid = np.zeros((len(uprime0_vals), len(zprime_vals)), dtype=float)
    c_grid = np.zeros((len(uprime0_vals), len(zprime_vals)), dtype=float)

    for i, uprime0 in enumerate(uprime0_vals):
        for j, zprime0 in enumerate(zprime_vals):
            res = solve_bvp_superposition(
                h, uprime0=float(uprime0), zprime0=float(zprime0)
            )
            max_err_grid[i, j] = max(res.abs_err)
            c_grid[i, j] = res.C

    plt.figure()
    plt.title("Assumption sensitivity (max |error|)")
    x = np.arange(len(zprime_vals) + 1)
    y = np.arange(len(uprime0_vals) + 1)
    min_pos = float(np.min(max_err_grid[max_err_grid > 0]))
    norm_err = colors.LogNorm(vmin=min_pos, vmax=float(np.max(max_err_grid)))
    im = plt.pcolormesh(x, y, max_err_grid, shading="flat", norm=norm_err)
    plt.colorbar(im, label="max |error|")
    plt.xticks(np.arange(len(zprime_vals)) + 0.5, [f"{v:g}" for v in zprime_vals])
    plt.yticks(np.arange(len(uprime0_vals)) + 0.5, [f"{v:g}" for v in uprime0_vals])
    plt.xlabel("zprime0")
    plt.ylabel("uprime0")
    for i, uprime0 in enumerate(uprime0_vals):
        for j, zprime0 in enumerate(zprime_vals):
            plt.text(
                j + 0.5,
                i + 0.5,
                f"{max_err_grid[i, j]:.6e}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    plt.grid(False)
    plt.savefig(
        out_dir / "report_prob3_assumption_max_error.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

    plt.figure()
    plt.title("Assumption sensitivity (C)")
    max_abs_c = float(np.max(np.abs(c_grid)))
    norm_c = colors.SymLogNorm(linthresh=1e-6, vmin=-max_abs_c, vmax=max_abs_c)
    im = plt.pcolormesh(x, y, c_grid, shading="flat", norm=norm_c)
    plt.colorbar(im, label="C")
    plt.xticks(np.arange(len(zprime_vals)) + 0.5, [f"{v:g}" for v in zprime_vals])
    plt.yticks(np.arange(len(uprime0_vals)) + 0.5, [f"{v:g}" for v in uprime0_vals])
    plt.xlabel("zprime0")
    plt.ylabel("uprime0")
    for i, uprime0 in enumerate(uprime0_vals):
        for j, zprime0 in enumerate(zprime_vals):
            plt.text(
                j + 0.5,
                i + 0.5,
                f"{c_grid[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    plt.grid(False)
    plt.savefig(
        out_dir / "report_prob3_assumption_c_grid.png", dpi=150, bbox_inches="tight"
    )
    plt.show()


def plot_result(
    res: BvpResult, *, title: str = "Superposition + RK4", show_exact: bool = False
) -> None:
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).resolve().parent

    plt.figure()
    plt.title(title)
    plt.plot(res.xs, res.y_num, marker="o", label="numerical solution")
    if show_exact:
        plt.plot(res.xs, res.y_ex, linestyle="--", label="exact")
    plt.xlim(0.0, 1.0)
    plt.ylim(bottom=0.0)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_dir / "report_prob3_solution.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.title("Absolute error")
    plt.plot(res.xs, res.abs_err, marker="o")
    plt.xlim(0.0, 1.0)
    plt.ylim(bottom=0.0)
    plt.xlabel("x")
    plt.ylabel("|error|")
    plt.grid(True)
    plt.show()


def main() -> None:
    # 刻み幅の選択
    h: float = 0.1

    # 基準ケース（通常の仮定）
    res: BvpResult = solve_bvp_superposition(h, uprime0=0.0, zprime0=1.0)
    print_summary(res, label=f"baseline h={h:g}")
    plot_result(res, title=f"Superposition + RK4 (h={h:g})")

    # 仮定の影響を確認
    uprime0_vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    zprime0_vals = [0.01, 0.1, 1.0, 10.0, 100.0]
    compare_assumptions(
        h,
        uprime0_list=uprime0_vals,
        zprime_list=zprime0_vals,
    )

    # 描画
    plot_assumption_grids(
        h,
        uprime0_list=uprime0_vals,
        zprime_list=zprime0_vals,
    )


if __name__ == "__main__":
    main()
