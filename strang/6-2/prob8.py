from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.integrate import solve_ivp
import time


# ---------------------------------------------------------
# 1. 問題のセットアップ
#    行列 K (1Dラプラシアン) を構築する
# ---------------------------------------------------------
def setup_problem(N):
    # n: 格子点の数 (問題文の n=2N+1)
    n = 2 * N + 1

    # dx: 格子間隔 (区間 [-1, 1] と仮定すると 2/(n+1) だが、ここでは単純化のため 1/(n+1) とする)
    # 重要: 物理的な拡散を再現するには、1/dx^2 のスケーリングが必要
    dx = 1.0 / (n + 1)

    # --- [Blank 1] 疎行列 K の構築 ---
    # K = (1/dx^2) * tridiag(-1, 2, -1)
    # これが「拡散（隣との平均化）」を表すと同時に、「硬さ（dx^2の逆数）」の源です。
    diag_lower = -1 * np.ones(n - 1)
    diag_main = 2 * np.ones(n)
    diag_upper = -1 * np.ones(n - 1)

    # ヒント: 係数 (1 / dx**2) を忘れずに！これがないと拡散しません。
    K = (
        diags(diag_lower, -1, shape=(n, n))
        + diags(diag_main, 0, shape=(n, n))
        + diags(diag_upper, 1, shape=(n, n))
    ) / (dx**2)

    # 初期値: 真ん中だけ高い（デルタベクトル）
    u0 = np.zeros(n)
    u0[N] = 1

    return K, u0, dx


# ---------------------------------------------------------
# 2. 後退オイラー法 (Backward Euler)
#    式: (I + dt * K) u_{n+1} = u_n
#       f(u,t)=Ku　より
# ---------------------------------------------------------
def backward_euler_solve(K, u0, dt, t_span):
    t_values = np.arange(t_span[0], t_span[1], dt)
    u_curr = u0.copy()
    n = len(u0)

    # --- 連立方程式の行列 A を作る ---
    # 陰解法は「未来の値を逆算する」ため、Ax = b を解く必要がある
    # 解くべき式: (I + dt * K) * u_next = u_curr
    identity = eye(n)
    A = identity + K * dt
    A = A.tocsc()

    start_time = time.time()
    for _ in t_values[:-1]:
        # Ax = b を解いて時間を進める
        #   (疎行列用ソルバー spsolve を使うと高速)
        u_curr = spsolve(A, u_curr)

    elapsed = time.time() - start_time
    return np.asarray(u_curr), elapsed


# ---------------------------------------------------------
# 3. Scipyのソルバー比較 (RK45 vs BDF)
# ---------------------------------------------------------
def run_scipy_solver(K, u0, t_span, method_name):
    # u' = -Ku を定義
    def fun(t, u):
        return -K.dot(u)

    start_time = time.time()

    # --- 硬い方程式用のメソッド指定 ---
    # scipy.integrate.solve_ivp は、デフォルトで適応的ステップサイズ制御が常に働いている
    #   'RK45' は4次ルンゲクッタ（陽解法）
    #   'BDF' は ode15s 相当（陰解法）
    sol = solve_ivp(fun, t_span, u0, method=method_name)

    elapsed = time.time() - start_time
    return sol, elapsed


def relative_error(u, u_ref):
    denom = np.linalg.norm(u_ref)
    if denom == 0:
        return np.linalg.norm(u - u_ref)
    return np.linalg.norm(u - u_ref) / denom


# =========================================================
# メイン実行部
# =========================================================
# N = 100 (n=201) くらいで実験。
# 勇気があるなら N = 1000 (n=2001) に書き換えてみよう... RK45が止まります。
# N_val = 100
# N_val = 500
N_val = 1000
K_mat, u_init, dx_val = setup_problem(N_val)
t_end = 0.01  # 短い時間でOK

print(f"Problem Size n = {2 * N_val + 1}")
print(f"Stiffness estimate (lambda_max) ~ {4 / (dx_val**2):.2f}")
print("-" * 40)

# 1. Backward Euler (Implicit)
dt_be = 0.001
u_be, time_be = backward_euler_solve(K_mat, u_init, dt=dt_be, t_span=[0, t_end])
print(f"Backward Euler (Implicit): {time_be:.4f} sec")

# 2. RK45 (ode45 equivalent) - Explicit
sol_rk, time_rk = run_scipy_solver(K_mat, u_init, [0, t_end], "RK45")
u_rk = np.asarray(sol_rk.y[:, -1])
print(f"Scipy RK45 (Explicit)  : {time_rk:.4f} sec")

# 3. BDF (ode15s equivalent) - Implicit
sol_bdf, time_bdf = run_scipy_solver(K_mat, u_init, [0, t_end], "BDF")
u_bdf = np.asarray(sol_bdf.y[:, -1])
print(f"Scipy BDF (Implicit)   : {time_bdf:.4f} sec")

# 基準解（高精度BDF）
def fun(t, u):
    return -K_mat.dot(u)

ref_sol = solve_ivp(fun, [0, t_end], u_init, method="BDF", rtol=1e-10, atol=1e-12)
u_ref = np.asarray(ref_sol.y[:, -1])

err_be = relative_error(u_be, u_ref)
err_rk = relative_error(u_rk, u_ref)
err_bdf = relative_error(u_bdf, u_ref)

print(f"Relative error (Backward Euler): {err_be:.2e}")
print(f"Relative error (RK45)         : {err_rk:.2e}")
print(f"Relative error (BDF)          : {err_bdf:.2e}")

# プロット
fig = plt.figure(figsize=(10, 6))
x = np.linspace(-1, 1, len(u_init))
# plt.plot(x, u_init, "k--", label="Initial", alpha=0.3)
plt.plot(x, u_be, label="Backward Euler")
plt.plot(x, u_rk, label="RK45")
plt.plot(x, u_bdf, label="BDF")
plt.title(f"Diffusion at t={t_end} (N={N_val})")
plt.legend()
plt.grid()
output_path = Path(__file__).with_name("diffusion_comparison.png")
fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
plt.show()

# 計算時間の比較
times = np.array([time_be, time_rk, time_bdf])
errors = np.array([err_be, err_rk, err_bdf])
labels = ["Backward Euler", "RK45", "BDF"]

fig = plt.figure(figsize=(6, 4))
plt.bar(labels, times, color=["#4C78A8", "#F58518", "#54A24B"])
plt.yscale("log")
plt.ylabel("Computation Time (s)")
plt.title("Computation Time Comparison")
plt.grid(axis="y", which="both", linestyle=":", alpha=0.6)
output_path = Path(__file__).with_name("time_comparison.png")
fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
plt.show()

# 精度の比較
fig = plt.figure(figsize=(6, 4))
plt.bar(labels, errors, color=["#4C78A8", "#F58518", "#54A24B"])
plt.yscale("log")
plt.ylabel("Relative Error")
plt.title("Accuracy Comparison")
plt.grid(axis="y", which="both", linestyle=":", alpha=0.6)
output_path = Path(__file__).with_name("accuracy_comparison.png")
fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
plt.show()

# 適応ステップ幅の推移（RK45 / BDF / Backward Euler）
dt_history = np.diff(sol_rk.t)
dt_history_bdf = np.diff(sol_bdf.t)
steps_be = int(np.ceil(t_end / dt_be))
dt_history_be = np.full(steps_be, dt_be)

print(f"RK45のステップ数: {len(dt_history)}")
print(f"最小 dt: {np.min(dt_history):.2e}")
print(f"最大 dt: {np.max(dt_history):.2e}")
print(f"BDFのステップ数: {len(dt_history_bdf)}")
print(f"BDF最小 dt: {np.min(dt_history_bdf):.2e}")
print(f"BDF最大 dt: {np.max(dt_history_bdf):.2e}")

fig = plt.figure(figsize=(6, 4))
plt.boxplot([dt_history, dt_history_bdf, dt_history_be], tick_labels=["RK45", "BDF", "Backward Euler"])
plt.title("Step Size Distribution")
plt.ylabel("dt")
plt.yscale("log")
plt.grid(axis="y", linestyle=":", alpha=0.6)
output_path = Path(__file__).with_name("step_size_distribution.png")
fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
plt.show()
