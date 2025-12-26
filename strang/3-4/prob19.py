import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_poisson_square(N_terms=20):
    # 1. 空間を作る（0から1までの正方形）
    x = np.linspace(0, 1, 40)
    y = np.linspace(0, 1, 40)
    X, Y = np.meshgrid(x, y)

    # 解 u を入れる箱（最初はゼロ）
    U = np.zeros_like(X)

    # 2. 無限級数を計算する（N項まで足し合わせる）
    # 式：u = Σ Σ [16 / (π^4 * i * j * (i^2 + j^2))] * sin(iπx) * sin(jπy)
    for i in range(1, N_terms, 2):  # 奇数だけ (1, 3, 5...)
        for j in range(1, N_terms, 2):
            # 係数の計算（ここが一番めんどくさい部分）
            coef = 16 / (np.pi**4 * i * j * (i**2 + j**2))

            # 波を足していく
            term = coef * np.sin(i * np.pi * X) * np.sin(j * np.pi * Y)
            U += term

    return X, Y, U


# --- 計算実行 ---
X, Y, Z = solve_poisson_square()

# --- 3Dで表示（これが見たかった！） ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# 色付きの曲面プロット
surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

ax.set_title("Solution to Poisson Equation on a Square\n(-u_xx - u_yy = 1)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
