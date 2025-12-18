import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def solve_normal_modes():
    # rosscode.m のパラメータを使用
    M = np.array([[9.0, 0.0], [0.0, 1.0]])
    K = np.array([[81.0, -6.0], [-6.0, 6.0]])

    # 初期条件
    u_zero = np.array([1.0, 0.0])
    v_zero = np.array([0.0, 0.0])

    # 1. 一般化固有値問題を解く Kx = lambda Mx
    # scipy.linalg.eigh は "Hermitian" (対称行列) 用のソルバ
    # type=1 は A x = w B x 形式を指定
    evals, evecs = la.eigh(K, M, type=1)

    # evals: 固有値 (lambda = omega^2)
    # evecs: 固有ベクトル行列 (M直交している)

    omega = np.sqrt(evals)  # 固有振動数

    print(f"固有値 (lambda): {evals}")
    print(f"固有振動数 (omega): {omega}")
    print(f"固有ベクトル:\n{evecs}")

    # 2. 係数 A, B を求める
    # u(0) = sum(A_i * x_i), v(0) = sum(B_i * omega_i * x_i)
    # 行列形式だと: u_zero = V * A_coeffs, v_zero = V * (omega * B_coeffs)

    # Pythonの solve は "vectors \ uzero" に相当
    A_coeffs = la.solve(evecs, u_zero)

    # Bは今回 v_zero=0 なので全ゼロだが、一般形として記述
    B_coeffs = la.solve(evecs, v_zero) / omega

    # 3. 時間発展を計算
    t = np.linspace(0, 10, 500)

    # 各モードの振動を足し合わせる (ブロードキャスト計算)
    # coeffs(t) = A cos(wt) + B sin(wt)
    # shape: (2, 500)
    mode_contributions = A_coeffs[:, None] * np.cos(omega[:, None] * t) + B_coeffs[
        :, None
    ] * np.sin(omega[:, None] * t)

    # u(t) = Matrix_V @ vector_coeffs(t)
    u_t = evecs @ mode_contributions

    # プロット
    plt.figure(figsize=(10, 4))
    plt.plot(t, u_t[0, :], label="u1 (Mass 9)", color="green")
    plt.plot(t, u_t[1, :], label="u2 (Mass 1)", color="hotpink")  # 第2章カラー
    plt.title("Normal Modes Solution (Exact)")
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return u_t


# 実行
u_exact = solve_normal_modes()
