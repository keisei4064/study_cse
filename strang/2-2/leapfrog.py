import numpy as np


def leapfrog_scalar(n):
    """
    n: タイムステップ数 (2pi までを n 分割)
    """
    dt = 2 * np.pi / n

    # 初期値: u(0)=0, u(dt) = 近似的に 3*dt (sin(3t)の微分から)
    u_old = 0.0
    u_curr = 3.0 * dt

    # 記録用
    u_history = [u_old, u_curr]

    # ループ
    for i in range(2, n + 1):
        # Leapfrog formula: u_{n+1} - 2u_n + u_{n-1} = -9 * dt^2 * u_n
        u_next = 2 * u_curr - u_old - 9 * (dt**2) * u_curr

        # 更新
        u_old = u_curr
        u_curr = u_next
        u_history.append(u_curr)

    return u_curr, dt, u_history


# テスト: n=9（不安定？）と n=100（安定）を比較
print("\n--- Leapfrog Stability Test ---")

# n=9 (dt が大きすぎる場合 -> 2pi/9 * 3 > 2 ? )
u_bad, dt_bad, hist_bad = leapfrog_scalar(9)
print(f"n=9   (dt={dt_bad:.2f}): u(2pi) = {u_bad:.4f} (発散の可能性)")

# n=40 (安定)
u_good, dt_good, hist_good = leapfrog_scalar(40)
print(f"n=40  (dt={dt_good:.2f}): u(2pi) = {u_good:.4f} (正解は sin(6pi)=0)")

# 精度確認 C = n^2 * error
error = abs(u_good - 0)
C = (40**2) * error
print(f"Accuracy Constant C = {C:.4f} (2次精度の確認)")
