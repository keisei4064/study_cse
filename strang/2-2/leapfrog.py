# p.119にあるサンプルコード
# 式(17)の蛙飛び法 (Leapfrog Method) を実装
# 解く問題
#   u''(t)+9u(t)=0
# 初期条件
#   u(0)=0, u'(0)=3

# 安定条件は: dt <= 2/sqrt(9) = 2/3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def leapfrog_scalar(dt):
    """
    dt: タイムステップ幅 (2pi までを dt で進める)
    """
    n = int(round(2 * np.pi / dt))

    # 初期値: u(0)=0, u(dt) = 近似的に 3*dt (sin(3t)の微分から)
    u_old = 0.0  # t=0
    u_curr = 3.0 * dt  # t=1

    # 記録用
    u_history = [u_old, u_curr]

    # ループ
    for i in range(2, n + 1):
        # 式(17)
        u_next = 2 * u_curr - u_old - 9 * (dt**2) * u_curr

        # 更新
        u_old = u_curr
        u_curr = u_next
        u_history.append(u_curr)

    return u_curr, dt, u_history


# テスト: dtが大きい場合（不安定？）と小さい場合（安定）を比較
print("\n--- Leapfrog Stability Test ---")
print("解析解: u(2pi) = 0")
print("安定条件: dt <= 2/3 ≈ 0.6667\n")

# dt が小さい場合 (安定)
u_good, dt_good, hist_good = leapfrog_scalar(2 * np.pi / 40)
print(f"dt={dt_good:.2f}: u(2pi) = {u_good:.4f} (2次精度)")

# dt が大きすぎる場合 -> 2pi/9 * 3 > 2
u_bad, dt_bad, hist_bad = leapfrog_scalar(2 * np.pi / 9)
print(f"dt={dt_bad:.2f}: u(2pi) = {u_bad:.4f} (発散)")

# 可視化: 安定/不安定を別スケールで比較
t_good = np.arange(len(hist_good)) * dt_good
t_bad = np.arange(len(hist_bad)) * dt_bad

fig, (ax_good, ax_bad) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

ax_good.plot(t_good, hist_good, "o-", label=f"dt={dt_good:.2f} (stable)")
ax_good.axhline(0, color="k", linewidth=0.8)
ax_good.set_xlabel("t")
ax_good.set_ylabel("u(t)")
ax_good.set_title("Stable case")
ax_good.legend()

ax_bad.plot(t_bad, hist_bad, "o-", label=f"dt={dt_bad:.2f} (unstable)")
ax_bad.axhline(0, color="k", linewidth=0.8)
ax_bad.set_xlabel("t")
ax_bad.set_ylabel("u(t)")
ax_bad.set_title("Unstable case")
ax_bad.legend()

fig.suptitle("Leapfrog Method Stability (separate scales)")
fig.tight_layout()
output_dir = Path(__file__).resolve().parent
fig.savefig(output_dir / "leapfrog_stability.png", dpi=150)
plt.show()
