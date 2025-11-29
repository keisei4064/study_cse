from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x):
    # 解析解
    # u(x) = x + (1 - e^x) / (e - 1)
    return x + (1 - np.exp(x)) / (np.exp(1) - 1)


# $u_0, u_{n+1}$ はゼロだから、変数は $u_1 ... u_n$ だけでいい


def solve_prob19(n=4):
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1 - h, n)  # 内点

    # 行列パーツ
    I = np.eye(n)
    K = 2 * I - np.eye(n, k=1) - np.eye(n, k=-1)
    Delta0 = np.eye(n, k=1) - np.eye(n, k=-1)  # 中心差分
    Delta_plus = -I + np.eye(n, k=1)  # 前進差分 (u_{i+1} - u_i)
    # ※境界条件 u_{n+1}=0 なので、最後の行の u_{i+1} は消えてOK

    # --- A. 中心差分 ---
    # (-u'' + u' = 1) => (1/h^2 K + 1/2h Delta0) u = 1
    A_central = (1 / h**2) * K + (1 / (2 * h)) * Delta0
    u_central = np.linalg.solve(A_central, np.ones(n))

    # --- B. 前進差分 ---
    # (-u'' + u' = 1) => (1/h^2 K + 1/h Delta_plus) u = 1
    A_forward = (1 / h**2) * K + (1 / h) * Delta_plus
    u_forward = np.linalg.solve(A_forward, np.ones(n))

    return x, u_central, u_forward


# 計算実行
n = 4
h = 1 / (n + 1)
x_grid, u_cent, u_fwd = solve_prob19(n)
u_exact = exact_solution(x_grid)
err_cent = np.abs(u_cent - u_exact)
err_fwd = np.abs(u_fwd - u_exact)

# 結果表示
print(f"--- 結果 (n={n}) ---")
print(
    f"{'x':<5} | {'Exact':<10} | {'Central':<10} | {'Fwd':<10} | {'Err(Central)':<10} | {'Err(Fwd)':<10}"
)
for i in range(n):
    print(
        f"{x_grid[i]:.2f}  | {u_exact[i]:.5f}    | {u_cent[i]:.5f}    | {u_fwd[i]:.5f}    | {err_cent[i]:.3e}    | {err_fwd[i]:.3e}"
    )

# グラフ描画
x_plot = np.linspace(0, 1, 100)
plt.figure(figsize=(8, 5))
plt.plot(x_plot, exact_solution(x_plot), "k-", label="Exact Solution", alpha=0.5)
plt.plot(
    np.concatenate(([0], x_grid, [1])),
    np.concatenate(([0], u_cent, [0])),
    "bo--",
    label="Central (2nd order)",
)
plt.plot(
    np.concatenate(([0], x_grid, [1])),
    np.concatenate(([0], u_fwd, [0])),
    "rs--",
    label="Forward (1st order)",
)
plt.title(f"Problem 19: -u'' + u' = 1 (n={n}, h={h})")
plt.legend()
plt.grid()
save_dir = Path(__file__).resolve().parent
plt.savefig(save_dir / f"prob19_n={n}_h={h}.png", dpi=200, bbox_inches="tight")
plt.show()
