from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def make_K1d(n: int) -> FloatArray:
    """1次元Poisson方程式の剛性行列K_nを作る.

    Args:
        n: 行列サイズ(未知数の数)。
    """
    A: FloatArray = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(A, 2.0)
    idx: NDArray[np.int64] = np.arange(n - 1, dtype=np.int64)
    A[idx, idx + 1] = -1.0
    A[idx + 1, idx] = -1.0
    return A


def rel_residual(A: FloatArray, x: FloatArray, b: FloatArray, r0_norm: float) -> float:
    """相対残差||r_k||/||r_0||を計算する.

    Args:
        A: 係数行列。
        x: 現在の近似解。
        b: 右辺ベクトル。
        r0_norm: 初期残差のノルム||r_0||。
    """
    r: FloatArray = b - A @ x
    return float(np.linalg.norm(r) / r0_norm)


@dataclass(frozen=True)
class IterResult:
    """反復法の履歴(近似解と相対残差)をまとめるコンテナ."""

    name: str
    xs: List[FloatArray]
    rel_res: List[float]


def jacobi(A: FloatArray, b: FloatArray, x0: FloatArray, iters: int) -> IterResult:
    """Jacobi法でAx=bを反復解法する.

    Args:
        A: 係数行列。
        b: 右辺ベクトル。
        x0: 初期値。
        iters: 反復回数。
    """
    D: FloatArray = np.diag(A).copy()
    R: FloatArray = A - np.diag(D)
    x: FloatArray = x0.copy()
    r0_norm: float = float(np.linalg.norm(b - A @ x))
    xs: List[FloatArray] = [x.copy()]
    rels: List[float] = [1.0]

    for _ in range(iters):
        x = (b - R @ x) / D
        xs.append(x.copy())
        rels.append(rel_residual(A, x, b, r0_norm))

    return IterResult("Jacobi", xs, rels)


def gauss_seidel(
    A: FloatArray, b: FloatArray, x0: FloatArray, iters: int
) -> IterResult:
    """Gauss-Seidel法でAx=bを反復解法する.

    Args:
        A: 係数行列。
        b: 右辺ベクトル。
        x0: 初期値。
        iters: 反復回数。
    """
    n: int = A.shape[0]
    x: FloatArray = x0.copy()
    r0_norm: float = float(np.linalg.norm(b - A @ x))
    xs: List[FloatArray] = [x.copy()]
    rels: List[float] = [1.0]

    # 標準的な前進GSスイープ(更新済み成分を即時利用)
    for _ in range(iters):
        for i in range(n):
            # ここでは三重対角だが一般形で書く(密行列なら1反復でO(n^2)。
            # K_nの三重対角なら実質O(n))。
            s1: float = float(A[i, :i] @ x[:i])  # 更新済み成分を使う
            s2: float = float(A[i, i + 1 :] @ x[i + 1 :])  # まだ更新していない成分
            x[i] = (b[i] - s1 - s2) / A[i, i]
        xs.append(x.copy())
        rels.append(rel_residual(A, x, b, r0_norm))

    return IterResult("Gauss-Seidel", xs, rels)


def sor(
    A: FloatArray, b: FloatArray, x0: FloatArray, iters: int, omega: float
) -> IterResult:
    """SOR法(緩和付きGS)でAx=bを反復解法する.

    Args:
        A: 係数行列。
        b: 右辺ベクトル。
        x0: 初期値。
        iters: 反復回数。
        omega: 緩和パラメータ(ω)。
    """
    n: int = A.shape[0]
    x: FloatArray = x0.copy()
    r0_norm: float = float(np.linalg.norm(b - A @ x))
    xs: List[FloatArray] = [x.copy()]
    rels: List[float] = [1.0]

    # 前進SORスイープ
    for _ in range(iters):
        for i in range(n):
            s1: float = float(A[i, :i] @ x[:i])  # 更新済み成分
            s2: float = float(A[i, i + 1 :] @ x[i + 1 :])  # 未更新成分
            x_gs: float = (b[i] - s1 - s2) / A[i, i]  # GS更新値
            x[i] = (1.0 - omega) * x[i] + omega * x_gs
        xs.append(x.copy())
        rels.append(rel_residual(A, x, b, r0_norm))

    return IterResult(f"SOR (omega={omega:.4f})", xs, rels)


def theory_rates_1d(n: int) -> Dict[str, float]:
    """1次元Poissonに対する理論的収束率(スペクトル半径)を返す.

    Args:
        n: 行列サイズ(未知数の数)。
    """
    theta: float = float(np.pi / (n + 1))
    rho_j: float = float(np.cos(theta))
    rho_gs: float = float(np.cos(theta) ** 2)
    rho_sor_opt: float = float(
        (1.0 - np.sin(theta)) / (1.0 + np.sin(theta))
    )  # 演習に与えられている式
    omega_opt: float = float(
        2.0 / (1.0 + np.sin(theta))
    )  # このモデルの標準的最適値
    return {
        "theta": theta,
        "rho_J": rho_j,
        "rho_GS": rho_gs,
        "rho_SOR_opt": rho_sor_opt,
        "omega_opt": omega_opt,
    }


def main() -> None:
    """反復法の収束比較を実行するエントリポイント."""
    # 収束の差が見える程度のnを選ぶ
    n: int = 201
    A: FloatArray = make_K1d(n)

    rng: np.random.Generator = np.random.default_rng(0)
    b: FloatArray = rng.normal(size=(n,)).astype(np.float64)
    x0: FloatArray = np.zeros((n,), dtype=np.float64)

    iters: int = 200

    th: Dict[str, float] = theory_rates_1d(n)
    omega_opt: float = th["omega_opt"]

    results: List[IterResult] = [
        jacobi(A, b, x0, iters),
        gauss_seidel(A, b, x0, iters),
        sor(A, b, x0, iters, omega=omega_opt),
    ]

    # 理論値を出力
    print(f"n={n}")
    print(f"theta = pi/(n+1) = {th['theta']:.6f}")
    print(f"theory rho_J       = {th['rho_J']:.6f}")
    print(f"theory rho_GS      = {th['rho_GS']:.6f}")
    print(f"theory omega_opt   = {th['omega_opt']:.6f}")
    print(f"theory rho_SOR_opt = {th['rho_SOR_opt']:.6f}")
    print()

    # 所定の残差に到達する反復回数を比較
    targets: List[float] = [1e-2, 1e-4, 1e-6]
    for res in results:
        print(res.name)
        for t in targets:
            hit: int | None = None
            for k, rr in enumerate(res.rel_res):
                if rr <= t:
                    hit = k
                    break
            print(
                f"  reach {t:>7.0e}: {hit if hit is not None else 'not reached'} iters"
            )
        print(f"  final rel_res @ {iters}: {res.rel_res[-1]:.3e}")
        print()

    # 任意: 収束履歴を可視化
    try:
        import matplotlib.pyplot as plt  # type: ignore

        for res in results:
            plt.semilogy(res.rel_res, label=res.name)
        plt.xlabel("iteration k")
        plt.ylabel("||r_k|| / ||r_0||")
        plt.title(f"Convergence on 1D Poisson K_n (n={n})")
        plt.grid(True, which="both")
        plt.legend()
        plt.show()

    except Exception:
        pass


if __name__ == "__main__":
    main()
