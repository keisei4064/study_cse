from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


def visualize_k2d_construction():
    # ---------------------------------------------------------
    # PART 1: 小さなサイズで「数値」と「魔法」を実感する
    # N=3 (全体で 3x3=9変数)
    # ---------------------------------------------------------
    N = 3
    I = np.eye(N)
    # 1Dラプラシアン K (-1, 2, -1)
    K = np.diag([2] * N) + np.diag([-1] * (N - 1), k=1) + np.diag([-1] * (N - 1), k=-1)

    # 教科書の式 (3) p.282
    # K2D = I (x) K + K (x) I
    # Pythonでは np.kron(A, B)

    # Term 1: 縦方向の成分 (I (x) K) -> ブロック対角になる
    Term1 = np.kron(I, K)

    # Term 2: 横方向の成分 (K (x) I) -> 遠くの成分を作る
    Term2 = np.kron(K, I)

    # Sum: 2Dラプラシアン (4, -1, -1, -1, -1)
    K2D = Term1 + Term2

    # --- 可視化 (Heatmap) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 表示用のヘルパー関数
    def plot_matrix(ax, mat, title):
        cax = ax.matshow(mat, cmap="coolwarm", vmin=-1, vmax=4)
        for (i, j), z in np.ndenumerate(mat):
            if z != 0:  # 0以外を表示
                ax.text(
                    j,
                    i,
                    "{:0.0f}".format(z),
                    ha="center",
                    va="center",
                    color="white" if abs(z) > 2 else "black",
                )
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    plot_matrix(axes[0], Term1, r"Term 1: $I \otimes K$")
    plot_matrix(axes[1], Term2, r"Term 2: $K \otimes I$")
    plot_matrix(axes[2], K2D, r"Sum: $K2D$ (Target)")

    plt.suptitle(f"K2D Construction (N={N}, Matrix Size={N * N}x{N * N})", fontsize=16)
    plt.tight_layout()
    output_path = Path(__file__).with_name("k2d_construction.png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # PART 2: 大きなサイズで「構造」と「帯幅」を見る
    # N=10 (全体で 100x100変数)
    # ---------------------------------------------------------
    N_large = 10
    # スパース行列として作成（メモリ節約）
    I_large = sp.eye(N_large)
    K_large = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N_large, N_large))

    K2D_large = sp.kron(I_large, K_large) + sp.kron(K_large, I_large)

    fig = plt.figure(figsize=(6, 6))
    plt.spy(K2D_large, markersize=2, color="black")
    plt.title(
        f"Sparsity Pattern of K2D (N={N_large}, Size=100x100)\nNon-zeros shown in black",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    output_path = Path(__file__).with_name("k2d_sparsity_pattern.png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    # 対角線から離れた「ストライプ」に注目！
    plt.show()


if __name__ == "__main__":
    visualize_k2d_construction()
