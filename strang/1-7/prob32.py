from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 4x5の前進差分行列 A を定義
# (各行が -1, 1 のペアを持つ階差行列)
A = np.array([[-1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, 1]])

print("=== 行列 A (4x5) ===")
print(A)

# 2. 擬似逆行列 A_pinv を計算 (SVDベース)
A_pinv = np.linalg.pinv(A)

print("\n=== 擬似逆行列 A^+ (5x4) ===")
# 見やすいように小数第3位で丸めます
print(np.round(A_pinv, 3))

# 3. 検証1: AA^+ (右から掛ける) -> 単位行列になるはず
AA_pinv = A @ A_pinv
print("\n=== 検証1: AA^+ (4x4) ===")
print(np.round(AA_pinv, 3))


# 4. 検証2: A^+A (左から掛ける) -> 単位行列にならないはず（射影行列）
A_pinv_A = A_pinv @ A
print("\n=== 検証2: A^+A (5x5) ===")
print(np.round(A_pinv_A, 3))


# --- 可視化（ヒートマップ） ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Forward difference matrix and pseudoinverse heatmaps", fontsize=14)

sns.heatmap(
    A, annot=True, fmt=".0f", cmap="coolwarm", cbar=False, ax=axes[0, 0], square=True
)
axes[0, 0].set_title(r"$A\, (4\times 5)$")

sns.heatmap(
    A_pinv,
    annot=True,
    fmt=".2f",
    cmap="Purples",
    cbar=False,
    ax=axes[0, 1],
    square=True,
)
axes[0, 1].set_title(r"$A^{+}\, (5\times 4)$")

sns.heatmap(
    AA_pinv, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=axes[1, 0], square=True
)
axes[1, 0].set_title(r"$AA^{+}\, (4\times 4)$")

sns.heatmap(
    A_pinv_A, annot=True, fmt=".2f", cmap="Reds", cbar=False, ax=axes[1, 1], square=True
)
axes[1, 1].set_title(r"$A^{+}A\, (5\times 5)$")

out_path = Path(__file__).parent / "prob32_heatmap.png"
plt.savefig(out_path, dpi=300)
print(f"Saved heatmap to: {out_path}")

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()
