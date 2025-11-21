from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def solve_potential_flow_cube(
    nx: int = 21,
    ny: int = 21,
    nz: int = 21,
    dx: float = 0.1,
    dy: float = 0.1,
    dz: float = 0.1,
    omega: float = 1.5,
    max_iter: int = 1000,
    tol: float = 1.0e-5,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    立方体周りの 3D ポテンシャル流れ（Laplace方程式）を
    SOR 法で解く。FORTRAN コードをほぼそのまま移植。

    Returns
    -------
    p, u, v, w : (nx, ny, nz) の配列
        p: ポテンシャル
        u, v, w: 速度成分
    """

    # ---- 配列確保（p, u, v, w 全部 0 初期化）----
    p: NDArray[np.float64] = np.zeros((nx, ny, nz), dtype=np.float64)
    u: NDArray[np.float64] = np.zeros_like(p)
    v: NDArray[np.float64] = np.zeros_like(p)
    w: NDArray[np.float64] = np.zeros_like(p)

    # ---- 格子間隔と係数 ----
    dx2: float = dx * dx
    dy2: float = dy * dy
    dz2: float = dz * dz
    c0: float = 2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2

    # FORTRAN の ia=8, ib=14 を 0-based に変換 → 7..13
    ia, ib = 7, 13
    ja, jb = 7, 13
    ka, kb = 7, 13

    # ---- 入口・出口のポテンシャル ----
    # p(1,j,k)=0, p(if,j,k)=1
    p[0, :, :] = 0.0
    p[nx - 1, :, :] = 1.0

    # ---- SOR 反復 ----
    rmax1: float | None = None

    for na in range(1, max_iter + 1):
        rmax: float = 0.0

        # 奇数回: i=1..nx-2, j=0..ny-1, k=0..nz-1
        # 偶数回: 逆順でスイープ
        if na % 2 == 1:
            i_range = range(1, nx - 1)
            j_range = range(0, ny)
            k_range = range(0, nz)
        else:
            i_range = range(nx - 2, 0, -1)
            j_range = range(ny - 1, -1, -1)
            k_range = range(nz - 1, -1, -1)

        for i in i_range:
            for j in j_range:
                for k in k_range:
                    # 立方体内部はスキップ
                    if (ia < i < ib) and (ja < j < jb) and (ka < k < kb):
                        continue

                    im1: int = i - 1
                    ip1: int = i + 1
                    jm1: int = j - 1
                    jp1: int = j + 1
                    km1: int = k - 1
                    kp1: int = k + 1

                    # ---- 外部境界 (y,z) の Neumann 条件 ----
                    if j == 0:
                        jm1 = 1
                    if j == ny - 1:
                        jp1 = ny - 2
                    if k == 0:
                        km1 = 1
                    if k == nz - 1:
                        kp1 = nz - 2

                    # ---- 立方体表面の境界条件 (x 面) ----
                    if (ja <= j <= jb) and (ka <= k <= kb):
                        if i == ib:
                            im1 = im1 + 2
                        if i == ia:
                            ip1 = ip1 - 2

                    # ---- 立方体表面 (y 面) ----
                    if (ia <= i <= ib) and (ka <= k <= kb):
                        if j == jb:
                            jm1 = jm1 + 2
                        if j == ja:
                            jp1 = jp1 - 2

                    # ---- 立方体表面 (z 面) ----
                    if (ia <= i <= ib) and (ja <= j <= jb):
                        if k == kb:
                            km1 = km1 + 2
                        if k == ka:
                            kp1 = kp1 - 2

                    # ---- Laplace 残差 r = p_xx + p_yy + p_zz ----
                    r: float = (
                        (p[im1, j, k] - 2.0 * p[i, j, k] + p[ip1, j, k]) / dx2
                        + (p[i, jm1, k] - 2.0 * p[i, j, k] + p[i, jp1, k]) / dy2
                        + (p[i, j, km1] - 2.0 * p[i, j, k] + p[i, j, kp1]) / dz2
                    )

                    # ---- SOR 更新 ----
                    p[i, j, k] = p[i, j, k] + omega * r / c0
                    if abs(r) > rmax:
                        rmax = abs(r)

        # 残差の正規化（1ステップ目の rmax で割る）
        if rmax1 is None:
            rmax1 = rmax
        rnorm: float = rmax / rmax1 if rmax1 != 0.0 else 0.0

        # ログを出したければここで print
        # print(f"na={na:3d}, rmax={rnorm:10.7f}")

        # 収束判定
        if (na < max_iter) and (rnorm > tol):
            continue
        break

    # ---- 速度計算 u,v,w = grad(p) ----
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 立方体内部はスキップ
                if (ia < i < ib) and (ja < j < jb) and (ka < k < kb):
                    continue

                dx1: float = 2.0 * dx
                dy1: float = 2.0 * dy
                dz1: float = 2.0 * dz

                im1: int = i - 1
                ip1: int = i + 1
                jm1: int = j - 1
                jp1: int = j + 1
                km1: int = k - 1
                kp1: int = k + 1

                # ---- x 方向外部境界 ----
                if i == 0:
                    im1 = 0
                    dx1 = dx
                if i == nx - 1:
                    ip1 = nx - 1
                    dx1 = dx

                # 立方体 x 面
                if (ja <= j <= jb) and (ka <= k <= kb):
                    if i == ib:
                        im1 = im1 + 1
                        dx1 = dx
                    if i == ia:
                        ip1 = ip1 - 1
                        dx1 = dx

                # ---- y 方向外部境界 ----
                if j == 0:
                    jm1 = 0
                    dy1 = dy
                if j == ny - 1:
                    jp1 = ny - 1
                    dy1 = dy

                # 立方体 y 面
                if (ia <= i <= ib) and (ka <= k <= kb):
                    if j == jb:
                        jm1 = jm1 + 1
                        dy1 = dy
                    if j == ja:
                        jp1 = jp1 - 1
                        dy1 = dy

                # ---- z 方向外部境界 ----
                if k == 0:
                    km1 = 0
                    dz1 = dz
                if k == nz - 1:
                    kp1 = nz - 1
                    dz1 = dz

                # 立方体 z 面
                if (ia <= i <= ib) and (ja <= j <= jb):
                    if k == kb:
                        km1 = km1 + 1
                        dz1 = dz
                    if k == ka:
                        kp1 = kp1 - 1
                        dz1 = dz

                u[i, j, k] = (p[ip1, j, k] - p[im1, j, k]) / dx1
                v[i, j, k] = (p[i, jp1, k] - p[i, jm1, k]) / dy1
                w[i, j, k] = (p[i, j, kp1] - p[i, j, km1]) / dz1

    return p, u, v, w


if __name__ == "__main__":
    phi, u, v, w = solve_potential_flow_cube()
    print("phi shape:", phi.shape)
    print(
        "max |u|,|v|,|w|:",
        float(np.max(np.abs(u))),
        float(np.max(np.abs(v))),
        float(np.max(np.abs(w))),
    )

    import matplotlib.pyplot as plt

    nx, ny, nz = phi.shape
    dx = dy = dz = 0.1
    x = np.linspace(0, dx * (nx - 1), nx)
    y = np.linspace(0, dy * (ny - 1), ny)
    z = np.linspace(0, dz * (nz - 1), nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # --- 3D slice visualization of phi ---
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    midx, midy, midz = nx // 2, ny // 2, nz // 2
    # insert three orthogonal filled contours as slices
    ax1.contourf(
        X[:, :, midz], Y[:, :, midz], phi[:, :, midz], zdir="z", offset=z[midz], levels=20
    )
    ax1.contourf(
        X[:, midy, :], Z[:, midy, :], phi[:, midy, :], zdir="y", offset=y[midy], levels=20
    )
    ax1.contourf(
        Y[midx, :, :], Z[midx, :, :], phi[midx, :, :], zdir="x", offset=x[midx], levels=20
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("3D Slices of Potential (phi)")
    plt.show()

    # --- 3D quiver of velocities (subsampled & excluding cube interior) ---
    step = 3
    ii = np.arange(0, nx, step)
    jj = np.arange(0, ny, step)
    kk = np.arange(0, nz, step)
    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing="ij")

    mask = ~((7 < II) & (II < 13) & (7 < JJ) & (JJ < 13) & (7 < KK) & (KK < 13))

    Xq = X[II, JJ, KK][mask]
    Yq = Y[II, JJ, KK][mask]
    Zq = Z[II, JJ, KK][mask]
    Uq = u[II, JJ, KK][mask]
    Vq = v[II, JJ, KK][mask]
    Wq = w[II, JJ, KK][mask]

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.quiver(Xq, Yq, Zq, Uq, Vq, Wq, length=0.05, normalize=True)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("3D Velocity Field (quiver)")
    plt.show()
