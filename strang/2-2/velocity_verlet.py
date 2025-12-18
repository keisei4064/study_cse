import numpy as np
import scipy.linalg as la


def run_velocity_verlet():
    # パラメータ定義
    M = np.array([[9.0, 0.0], [0.0, 1.0]])
    K = np.array([[81.0, -6.0], [-6.0, 6.0]])

    # Mの逆行列（計算高速化のため。実際はsolveを使う方が安全な場合もある）
    M_inv = la.inv(M)

    # 加速度を計算する関数 a = -M^{-1} K u
    def get_accel(u_vec):
        return -M_inv @ (K @ u_vec)

    # 収束確認用リスト
    results_u = []
    results_v = []
    dt_list = []

    print(f"{'dt':^10} | {'u1 (End)':^12} | {'u2 (End)':^12}")
    print("-" * 40)

    # j=0 から 8 までループ（dtを半分にしていく）
    for j in range(9):
        dt = 0.63 / (2**j)
        dt_list.append(dt)

        # 初期値
        u = np.array([1.0, 0.0])
        v = np.array([0.0, 0.0])

        # 終了時刻 t_end = 63 (MATLABコード準拠)
        steps = int(100 * (2**j))

        # 初回の加速度
        a = get_accel(u)

        # メインループ (Velocity Verlet)
        for _ in range(steps):
            # 1. 半ステップ速度更新
            v_half = v + 0.5 * dt * a

            # 2. 位置更新
            u = u + dt * v_half

            # 3. 新しい加速度
            a_new = get_accel(u)

            # 4. 残りの半ステップ速度更新
            v = v_half + 0.5 * dt * a_new

            # 加速度更新
            a = a_new

        results_u.append(u)
        results_v.append(v)

        print(f"{dt:.5f} | {u[0]:12.4f} | {u[1]:12.4f}")

    return np.array(results_u).T


# 実行
U_final = run_velocity_verlet()
