import numpy as np
import matplotlib.pyplot as plt


def poisson_solver(x, y=None, N=39):
    """
    ストラング本 p.272 poisson.m のPython移植版

    Parameters:
        x (ndarray): x座標の配列（またはメッシュグリッド）
        y (ndarray): y座標の配列（またはメッシュグリッド）。Noneの場合は1次元計算。
        N (int): 級数の打ち切り項数（デフォルト39）

    Returns:
        u (ndarray): ポテンシャル場
    """

    # --- 1次元の場合 (変数がxのみ) ---
    if y is None:
        u = np.zeros_like(x)
        # MATLAB: for k = 1:2:N (1からNまでの奇数)
        for k in range(1, N + 1, 2):
            # 係数: 4 / (pi^3 * k^3)
            # MATLAB: 2^2/pi^3/k^3
            coef = 4.0 / (np.pi**3 * k**3)
            term = coef * np.sin(k * np.pi * x)
            u += term
        return u

    # --- 2次元の場合 (変数xとyがある) ---
    else:
        # 正方形領域で -uxx -uyy = 1 を解く
        u = np.zeros_like(x)  # xと同じ形状のゼロ配列を作成

        # MATLAB: for i = 1:2:N, for j = 1:2:N
        for i in range(1, N + 1, 2):
            for j in range(1, N + 1, 2):
                # 係数: 16 / (pi^4 * i * j * (i^2 + j^2))
                # MATLAB: 2^4/pi^4/(i*j)/(i^2+j^2)
                numerator = 16.0
                denominator = (np.pi**4) * i * j * (i**2 + j**2)
                coef = numerator / denominator

                term = coef * np.sin(i * np.pi * x) * np.sin(j * np.pi * y)
                u += term
        return u


# ==========================================
# 視覚化パート
# ==========================================

# 設定
N_terms = 39  # 級数の項数

plt.figure(figsize=(12, 5))

# --- ケース1: 1次元のプロット ---
plt.subplot(1, 2, 1)

# x座標: 0から1まで (MATLAB: xx = 0:.01:1)
xx_1d = np.linspace(0, 1, 101)
uu_1d = poisson_solver(xx_1d, N=N_terms)

plt.plot(xx_1d, uu_1d, "b-", linewidth=2)
plt.title(f"1D Solution (N={N_terms})")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)


# --- ケース2: 2次元のプロット (等高線図) ---
plt.subplot(1, 2, 2)

# メッシュグリッドの作成 (MATLAB: meshgrid(0:.1:1, 0:.1:1))
# 少し滑らかにするため分割数を増やしています (50x50)
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
XX, YY = np.meshgrid(x, y)

# 2次元解の計算
ZZ = poisson_solver(XX, YY, N=N_terms)

# 等高線図のプロット (MATLAB: contourf)
contour = plt.contourf(XX, YY, ZZ, 20, cmap="viridis")
plt.colorbar(contour, label="u(x, y)")
plt.title(f"2D Solution (Poisson Equation) on Unit Square")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("square")  # アスペクト比を1:1に

plt.tight_layout()
plt.show()
