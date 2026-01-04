"""
ヤコビ法（Jacobi eigenvalue algorithm）
- 対称行列 B の固有値を、1ステップ（1回転）ずつ収束まで計算
- 解析解（与えられた固有値）と比較
- 性能評価：オフ対角成分ノルムの減少、反復回数など

行列 B:
    [[1, 1, 4, 0],
     [1, 5, 1, 0],
     [4, 1, 1, 0],
     [0, 0, 0, 2]]

解析解:
    λ = -3, 2, 3.5858, 6.4142
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator


FloatVec = NDArray[np.float64]
FloatMat = NDArray[np.float64]

TOL: float = 1e-10
MAX_ITERS: int = 20


@dataclass(frozen=True)
class JacobiStep:
    """ヤコビ法の1ステップの記録データ"""

    it: int  # 反復回数（1始まり）
    p: int  # ピボット位置の行インデックス
    q: int  # ピボット位置の列インデックス
    a_pq_before: float  # 回転前の A[p,q]
    offdiag_fro: float  # オフ対角成分のフロベニウスノルム
    max_offdiag: float  # 最大のオフ対角 |A[p,q]|


def make_matrix_B() -> FloatMat:
    # 問題の行列 B
    B: FloatMat = np.array(
        [
            [1.0, 1.0, 4.0, 0.0],
            [1.0, 5.0, 1.0, 0.0],
            [4.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    return B


def analytic_eigenvalues() -> FloatVec:
    # 固有値の解析解
    return np.array([-3.0, 2.0, 3.5858, 6.4142], dtype=np.float64)


def offdiag_frobenius_norm(A: FloatMat) -> float:
    """非対角成分のフロベニウスノルム ||A - diag(A)||_F"""
    # ||A - diag(A)||_F
    D = np.diag(np.diag(A))
    return float(np.linalg.norm(A - D, ord="fro"))


def max_offdiag_abs(A: FloatMat) -> tuple[float, int, int]:
    """最大の非対角成分 |A[i,j]| (i<j) と、その位置 (p,q) を返す"""

    # 上三角部分を走査して最大値探索
    n = A.shape[0]
    p, q = 0, 1
    m = abs(float(A[p, q]))
    for i in range(n):
        for j in range(i + 1, n):
            v = abs(float(A[i, j]))
            if v > m:
                m = v
                p, q = i, j
    return m, p, q


def jacobi_rotation_params(
    a_pp: float, a_qq: float, a_pq: float
) -> tuple[float, float]:
    """
    対称行列の (p,q) 成分 a_pq を 0 にする回転の c=cosθ, s=sinθ を返す。

    授業資料の流れに合わせた式：
        tan(2θ) = 2 a_pq / (a_pp - a_qq)
        θ = 1/2 atan2(2 a_pq, a_pp - a_qq)
        c = cosθ, s = sinθ
    """
    if a_pq == 0.0:  # 非対角がゼロなら回転不要
        return 1.0, 0.0
    theta = 0.5 * math.atan2(2.0 * a_pq, a_pp - a_qq)
    c = math.cos(theta)
    s = math.sin(theta)
    return c, s


def apply_jacobi_rotation(A: FloatMat, V: FloatMat, p: int, q: int) -> None:
    """
    A <- G^T A G で A[p,q]=0 に近づける
    V <- V G で固有ベクトルも更新する
    """
    a_pp = float(A[p, p])
    a_qq = float(A[q, q])
    a_pq = float(A[p, q])

    c, s = jacobi_rotation_params(a_pp, a_qq, a_pq)

    n = A.shape[0]

    # A の更新（対称性を保つ）
    for k in range(n):  # 行方向に走査
        if k == p or k == q:
            # 対角成分は後で更新
            continue
        a_kp = float(A[k, p])  # p列
        a_kq = float(A[k, q])  # q列

        A[k, p] = c * a_kp - s * a_kq
        A[p, k] = A[k, p]  # 対称性

        A[k, q] = s * a_kp + c * a_kq
        A[q, k] = A[k, q]  # 対称性

    # 対角要素更新
    A[p, p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq
    A[q, q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq

    # 指定した非対角要素を明示的に 0
    A[p, q] = 0.0
    A[q, p] = 0.0

    # 固有ベクトル行列 V の更新：V <- V G
    for k in range(n):
        v_kp = float(V[k, p])
        v_kq = float(V[k, q])
        V[k, p] = c * v_kp - s * v_kq
        V[k, q] = s * v_kp + c * v_kq
    return


def jacobi_eigen(
    A0: FloatMat,
    tol: float,
    max_iters: int,
) -> tuple[
    FloatVec,
    FloatMat,
    list[JacobiStep],
    list[FloatVec],
    list[FloatMat],
    list[FloatMat],
    list[FloatMat],
]:
    """
    1回転=1ステップで収束まで回すヤコビ法。
    戻り値：
        - 固有値（対角成分）
        - 固有ベクトル行列 V（列が固有ベクトル）
      - 収束過程（性能評価用ログ）
      - 各反復の固有値推移（対角成分）
      - 各反復の行列（ヒートマップ用）
      - 各反復の回転行列 G（ヒートマップ用）
      - 各反復の固有ベクトル行列 V（ヒートマップ用）
    """
    A = A0.astype(np.float64, copy=True)
    n = A.shape[0]
    V = np.eye(n, dtype=np.float64)

    history: list[JacobiStep] = []
    evals_history: list[FloatVec] = []
    mats_history: list[FloatMat] = [A.copy()]
    rotations_history: list[FloatMat] = []
    vecs_history: list[FloatMat] = [V.copy()]

    for it in range(1, max_iters + 1):
        maxv, p, q = max_offdiag_abs(A)
        off_f = offdiag_frobenius_norm(A)
        evals_history.append(np.sort(np.diag(A).copy()))

        history.append(
            JacobiStep(
                it=it,
                p=p,
                q=q,
                a_pq_before=float(A[p, q]),
                offdiag_fro=off_f,
                max_offdiag=maxv,
            )
        )

        # 収束判定：最大の非対角成分が十分小さい
        if maxv < tol:
            break

        a_pp = float(A[p, p])
        a_qq = float(A[q, q])
        a_pq = float(A[p, q])
        c, s = jacobi_rotation_params(a_pp, a_qq, a_pq)
        G = np.eye(n, dtype=np.float64)
        G[p, p] = c
        G[q, q] = c
        G[p, q] = s
        G[q, p] = -s
        rotations_history.append(G)

        apply_jacobi_rotation(A, V, p, q)
        mats_history.append(A.copy())
        vecs_history.append(V.copy())

    evals = np.diag(A).copy()
    return (
        evals,
        V,
        history,
        evals_history,
        mats_history,
        rotations_history,
        vecs_history,
    )


def sort_eigs(evals: FloatVec, evecs: FloatMat) -> tuple[FloatVec, FloatMat]:
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]


def main() -> None:
    B = make_matrix_B()
    evals_true = analytic_eigenvalues()

    evals, evecs, hist, evals_hist, mats_hist, rots_hist, vecs_hist = jacobi_eigen(
        B, tol=TOL, max_iters=MAX_ITERS
    )
    evals, evecs = sort_eigs(evals, evecs)

    # 解析解もソート
    evals_true_sorted = np.sort(evals_true)

    print("=== Jacobi method result ===")
    print(f"tol = {TOL:g}, max_iters = {MAX_ITERS}")
    print(f"iterations = {hist[-1].it if hist else 0}")
    print()
    print("computed eigenvalues:")
    for v in evals.tolist():
        print(f"  {v:.12f}")
    print()
    print("analytic eigenvalues (given):")
    for v in evals_true_sorted.tolist():
        print(f"  {v:.12f}")
    print()

    # 比較（対応付け：ソート順でOK）
    diff = evals - evals_true_sorted
    print("abs error vs analytic (sorted pairing):")
    for i in range(len(evals)):
        print(f"  |Δλ[{i}]| = {abs(float(diff[i])):.3e}")

    # 性能評価のためのログ出力（最初の数行と最後）
    print()
    print("=== Convergence log (selected) ===")
    print(f"{'it':>5} | {'p':>2} {'q':>2} | {'|A[p,q]|':>10} | {'offdiag_F':>12}")
    print("-" * 46)
    show_head = min(10, len(hist))
    for r in hist[:show_head]:
        print(
            f"{r.it:>5d} | {r.p:>2d} {r.q:>2d} | {abs(r.a_pq_before):>10.3e} | {r.offdiag_fro:>12.3e}"
        )
    if len(hist) > show_head:
        r = hist[-1]
        print("  ...")
        print(
            f"{r.it:>5d} | {r.p:>2d} {r.q:>2d} | {abs(r.a_pq_before):>10.3e} | {r.offdiag_fro:>12.3e}"
        )

    ys = [h.offdiag_fro for h in hist]
    xs = list(range(1, len(ys) + 1))
    out_dir = Path(__file__).resolve().parent

    fig_offdiag = plt.figure()
    plt.semilogy(xs, ys)  # 対数で見ると収束の性質が分かる
    plt.xlabel(r"$i$")
    plt.ylabel("offdiag Frobenius norm")
    plt.title("Jacobi method convergence")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1, len(xs))
    fig_offdiag.savefig(out_dir / "jacobi_offdiag_convergence.png", dpi=150)

    ys_eigs = np.array(evals_hist)
    xs_eigs = list(range(1, len(ys_eigs) + 1))
    fig_eigs = plt.figure()
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, ys_eigs.shape[1]))
    for i in range(ys_eigs.shape[1]):
        plt.plot(
            xs_eigs,
            ys_eigs[:, i],
            color=colors[i],
            label=rf"$\lambda_{{{i + 1}}}(i)$",
        )
    for i, v in enumerate(evals_true_sorted.tolist()):
        plt.axhline(
            v,
            linestyle="--",
            color=colors[i],
            linewidth=1.0,
            label=rf"$\lambda_{{{i + 1}}}$ (analytic)",
        )
    plt.xlabel(r"$i$ (iteration step)")
    plt.ylabel(r"$\lambda$ (eigenvalue)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(1, len(xs_eigs))
    plt.gcf().subplots_adjust(right=0.7)
    fig_eigs.savefig(out_dir / "jacobi_eigen_convergence.png", dpi=150)

    fig, ax = plt.subplots()
    vmax = float(np.max(np.abs(mats_hist[0])))
    im = ax.imshow(mats_hist[0], vmin=-vmax, vmax=vmax, cmap="coolwarm")
    fig.colorbar(im, ax=ax)
    ax.set_title("B heatmap (i=0)")
    ax.set_xticks(range(mats_hist[0].shape[1]))
    ax.set_yticks(range(mats_hist[0].shape[0]))
    ax.set_xticklabels([str(i + 1) for i in range(mats_hist[0].shape[1])])
    ax.set_yticklabels([str(i + 1) for i in range(mats_hist[0].shape[0])])
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    texts: list[list[Text]] = []
    for i in range(mats_hist[0].shape[0]):
        row: list[Text] = []
        for j in range(mats_hist[0].shape[1]):
            row.append(
                ax.text(
                    j,
                    i,
                    f"{mats_hist[0][i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
            )
        texts.append(row)

    def update(frame: int):
        data = mats_hist[frame]
        im.set_data(data)
        ax.set_title(f"B heatmap (i={frame})")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                texts[i][j].set_text(f"{data[i, j]:.2f}")
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(mats_hist), interval=400, blit=False
    )
    ani.save(out_dir / "jacobi_B_heatmap.gif", writer="pillow", fps=2)

    n_frames = len(mats_hist)
    n_cols = max(n_frames, 2)
    fig2 = plt.figure(
        figsize=(2.8 * n_cols, 8.4),
        constrained_layout=True,
    )
    gs = fig2.add_gridspec(3, n_cols, height_ratios=[1, 1, 1])
    vmax2 = float(np.max(np.abs(mats_hist[0])))
    for i in range(n_frames):
        ax2 = fig2.add_subplot(gs[0, i])
        data2 = mats_hist[i]
        ax2.imshow(data2, vmin=-vmax2, vmax=vmax2, cmap="coolwarm")
        ax2.set_title(f"B (i={i})")
        ax2.set_xticks([])
        ax2.set_yticks([])
        if i == 0:
            ax2.set_ylabel("B")
        for r in range(data2.shape[0]):
            for c in range(data2.shape[1]):
                ax2.text(
                    c,
                    r,
                    f"{data2[r, c]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    g_mats = rots_hist[:2]
    vmax_g = 1.0
    for i, g_mat in enumerate(g_mats):
        axg = fig2.add_subplot(gs[1, i])
        axg.imshow(g_mat, vmin=-vmax_g, vmax=vmax_g, cmap="coolwarm")
        axg.set_title(f"G (i={i + 1})")
        axg.set_xticks([])
        axg.set_yticks([])
        if i == 0:
            axg.set_ylabel("rotation G")
        for r in range(g_mat.shape[0]):
            for c in range(g_mat.shape[1]):
                axg.text(
                    c,
                    r,
                    f"{g_mat[r, c]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    vmax_x = float(np.max(np.abs(vecs_hist[0])))
    for i in range(n_frames):
        axx = fig2.add_subplot(gs[2, i])
        x_mat = vecs_hist[i]
        axx.imshow(x_mat, vmin=-vmax_x, vmax=vmax_x, cmap="coolwarm")
        axx.set_title(f"X (i={i})")
        axx.set_xticks([])
        axx.set_yticks([])
        if i == 0:
            axx.set_ylabel("eigenvectors X")
        for r in range(x_mat.shape[0]):
            for c in range(x_mat.shape[1]):
                axx.text(
                    c,
                    r,
                    f"{x_mat[r, c]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    fig2.suptitle("B heatmaps (top), rotation G (middle), and eigenvectors X (bottom)")
    fig2.savefig(out_dir / "jacobi_B_G_X_grid.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
