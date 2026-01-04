"""
数値解析学：べき乗法（Power Method）で「絶対値最大固有値」に対応する固有ベクトルを
10ステップ（1ステップずつ）出して、解析解と比較＆性能評価する

- 問題の行列:
    A = [[ 2,  2, -1],
         [ 0, -1,  0],
         [ 0, -5,  3]]

- 解析解:
    λ = 3,  x = (1, 0, -1)^T
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatVec = NDArray[np.float64]
FloatMat = NDArray[np.float64]


USE_REFERENCE_IMPLEMENTATION: bool = False


@dataclass(frozen=True)
class StepResult:
    k: int
    x: FloatVec
    lam_rq: float
    residual_norm: float
    err_to_true: float


def make_problem_matrix() -> FloatMat:
    """問題の行列 A を返す"""
    A: FloatMat = np.array(
        [
            [2.0, 2.0, -1.0],
            [0.0, -1.0, 0.0],
            [0.0, -5.0, 3.0],
        ],
        dtype=np.float64,
    )
    return A


def analytic_solution() -> tuple[float, FloatVec]:
    """解析解 λ と x を返す"""
    lam_true: float = 3.0
    x_true: FloatVec = np.array([1.0, 0.0, -1.0], dtype=np.float64)
    return lam_true, x_true


def normalize_first_component_to_one(v: FloatVec) -> FloatVec:
    """v の第一成分が 1 になるようにスケールして返す"""
    if float(v[0]) == 0.0:
        raise ValueError("zero vector cannot be normalized")

    return v / float(v[0])


def rayleigh_quotient(A: FloatMat, x: FloatVec) -> float:
    """
    レイリー商（固有値近似）
        λ ≈ (x^T A x) / (x^T x)
    """
    num = float(x.T @ (A @ x))
    den = float(x.T @ x)
    if den == 0.0:
        raise ValueError("x has zero norm; cannot compute Rayleigh quotient")
    return float(num) / float(den)


def residual_norm(A: FloatMat, x: FloatVec, lam: float) -> float:
    """||A x - lam x||_2 : 2ノルムで評価"""
    r: FloatVec = (A @ x) - (lam * x)
    return float(np.linalg.norm(r, ord=2))


def error_to_true(x: FloatVec, x_true: FloatVec) -> float:
    """
    解析解 x_true（第一成分=1）との差のノルム
    """
    # どちらも第一成分=1に揃えてから比較する
    x_n: FloatVec = normalize_first_component_to_one(x.copy())
    t_n: FloatVec = normalize_first_component_to_one(x_true.copy())
    return float(np.linalg.norm(x_n - t_n, ord=2))


def power_method(
    A: FloatMat,
    x0: FloatVec,
    steps: int,
    x_true: FloatVec,
) -> list[StepResult]:
    """べき乗法のステップを steps 回実行し、その結果を返す"""
    x: FloatVec = normalize_first_component_to_one(x0.astype(np.float64, copy=True))
    results: list[StepResult] = []

    for k in range(1, steps + 1):
        y: FloatVec = A @ x
        x = normalize_first_component_to_one(y)

        lam: float = rayleigh_quotient(A, x)
        res: float = residual_norm(A, x, lam)
        err: float = error_to_true(x, x_true)

        results.append(
            StepResult(k=k, x=x.copy(), lam_rq=lam, residual_norm=res, err_to_true=err)
        )

    return results


def format_vec(v: FloatVec, ndigits: int = 6) -> str:
    fmt = f"{{:+.{ndigits}f}}"
    return "[" + ", ".join(fmt.format(float(a)) for a in v.tolist()) + "]"


def format_vec_min(v: FloatVec) -> str:
    def fmt_val(val: float) -> str:
        if float(val).is_integer():
            return f"{int(val)}"
        return f"{val:g}"

    return "[" + ", ".join(fmt_val(float(a)) for a in v.tolist()) + "]"


def print_report(
    A: FloatMat,
    lam_true: float,
    x_true: FloatVec,
    results: Iterable[StepResult],
    label: str,
) -> None:
    print("=== Problem ===")
    print("A =")
    print(A)
    print(f"analytic λ = {lam_true}")
    print(f"analytic x (x[0]=1) = {format_vec(x_true)}")
    print()

    print(f"=== Power Method (step-by-step): {label} ===")
    header = f"{'k':>2} | {'x (x[0]=1)':>34} | {'Rayleigh λ':>11} | {'||Ax-λx||2':>12} | {'||x-x*||2':>10}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.k:>2d} | {format_vec(r.x):>34} | {r.lam_rq:>11.8f} | {r.residual_norm:>12.3e} | {r.err_to_true:>10.3e}"
        )

    # # 性能評価の一例（収束率っぽい指標）
    # # TODO: 本気で評価するなら「λ2/λ1（スペクトルギャップ）」や残差の減衰率を見る
    # res_list: list[float] = [r.residual_norm for r in results]
    # if len(res_list) >= 2 and res_list[0] > 0.0:
    #     ratios = [
    #         res_list[i] / res_list[i - 1]
    #         for i in range(1, len(res_list))
    #         if res_list[i - 1] > 0.0
    #     ]
    #     if ratios:
    #         print()
    #         print("=== Rough performance hint ===")
    #         print(f"residual ratio (last) ≈ {ratios[-1]:.6f}")
    #         print(
    #             "※ これをどう解釈するかは自分で説明を書いて（例：一定比で減ってる/減ってない等）"
    #         )


def write_results_tsv(
    path: Path,
    results: Iterable[StepResult],
    x_true: FloatVec,
    lam_true: float,
) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("k\tx_k\tx\t||x_k-x||\tlambda_k\tlambda\t|lambda_k-lambda|\n")
        for r in results:
            x_vec = format_vec(r.x, ndigits=3)
            x_true_vec = format_vec_min(x_true)
            lam_err = abs(r.lam_rq - lam_true)
            f.write(
                f"{r.k}\t{x_vec}\t{x_true_vec}\t{r.err_to_true:.3f}\t"
                f"{r.lam_rq:.3f}\t{lam_true:g}\t{lam_err:.3f}\n"
            )


def plot_residual_ratio(
    path: Path, series: Iterable[tuple[str, list[StepResult]]]
) -> None:
    fig, ax = plt.subplots()
    for label, results in series:
        res_list = [r.residual_norm for r in results]
        ks: list[int] = []
        ratios: list[float] = []
        for i in range(1, len(res_list)):
            if res_list[i - 1] == 0.0:
                continue
            ks.append(i + 1)
            ratios.append(res_list[i] / res_list[i - 1])
        if ks:
            ax.plot(ks, ratios, marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("$k$")
    ax.set_xlim(1, 10)
    ax.set_ylabel(r"$\|r_k\|/\|r_{k-1}\|$")
    ax.set_title("Residual Norm Ratio per Step")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


def plot_step_results(
    path: Path, series: Iterable[tuple[str, list[StepResult]]], lam_true: float
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    axes[0].set_xlabel("$k$")
    axes[0].set_xlim(1, 10)
    axes[0].set_ylabel(r"$|\lambda_k-\lambda|$")
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].set_xlabel("$k$")
    axes[1].set_xlim(1, 10)
    axes[1].set_ylabel(r"$\|x_k-x\|_2$")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    for label, results in series:
        ks = [r.k for r in results]
        lam_err = [abs(r.lam_rq - lam_true) for r in results]
        err = [r.err_to_true for r in results]
        axes[0].plot(ks, lam_err, marker="o", label=label)
        axes[1].plot(ks, err, marker="o", label=label)

    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)


def main() -> None:
    A = make_problem_matrix()
    lam_true, x_true = analytic_solution()

    steps = 10
    x0_list = [
        ("x0=[1,1,1]", "x0_1_1_1", np.array([1.0, 1.0, 1.0], dtype=np.float64)),
        ("x0=[1,0,1]", "x0_1_0_1", np.array([1.0, 0.0, 1.0], dtype=np.float64)),
        ("x0=[1,-1,1]", "x0_1_-1_1", np.array([1.0, -1.0, 1.0], dtype=np.float64)),
    ]

    series: list[tuple[str, list[StepResult]]] = []
    for label, suffix, x0 in x0_list:
        results = power_method(A=A, x0=x0, steps=steps, x_true=x_true)
        series.append((label, results))
        print_report(A=A, lam_true=lam_true, x_true=x_true, results=results, label=label)
        out_path = (
            Path(__file__).resolve().parent
            / f"report_prob4_power_method_{suffix}.tsv"
        )
        write_results_tsv(out_path, results, x_true, lam_true)

    plot_path = (
        Path(__file__).resolve().parent / "report_prob4_power_method_residual_ratio.png"
    )
    plot_residual_ratio(plot_path, series)
    steps_path = Path(__file__).resolve().parent / "report_prob4_power_method_steps.png"
    plot_step_results(steps_path, series, lam_true)


if __name__ == "__main__":
    main()
