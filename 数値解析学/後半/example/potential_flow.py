from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


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
    立方体周りの 3D ポテンシャル流れ（Laplace方程式）をSOR 法で解く

    Parameters
    ----------
    nx, ny, nz : int
        x, y, z 方向の格子点数
    dx, dy, dz : float
        x, y, z 方向の格子間隔
    omega : float
        SOR 緩和係数
    max_iter : int
        最大反復回数
    tol : float
        収束判定の許容誤差（正規化残差の閾値）

    Returns
    -------
    p, u, v, w : (nx, ny, nz) の配列
        p: ポテンシャル
        u, v, w: 速度成分
    """

    # ---- 配列確保（p を 0 初期化）----
    p: NDArray[np.float64] = np.zeros(
        (nx, ny, nz), dtype=np.float64
    )  # 速度ポテンシャル

    # ---- 格子間隔と係数 ----
    # dx2, dy2, dz2: 格子間隔の二乗
    dx2: float = dx * dx
    dy2: float = dy * dy
    dz2: float = dz * dz
    # L: 離散ラプラス方程式を整理した時に出る定数係数
    L: float = 2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2

    # 立方体（障害物）領域のインデックス範囲: x=ia~ib, y=ja~jb, z=ka~kb
    #   FORTRAN の ia=8, ib=14 を 0-based に変換 → 7..13
    ia, ib = 7, 13
    ja, jb = 7, 13
    ka, kb = 7, 13

    # ---- 入口・出口のポテンシャル ----
    # x=0 面を 0, x=最大面を 1 に固定するDirichlet条件
    #   p(1,j,k)=0, p(if,j,k)=1
    p[0, :, :] = 0.0
    p[-1, :, :] = 1.0

    # ---- SOR 反復 ----
    res_max0: float | None = None  # 正規化に使う初回反復の最大残差

    for iter_count in range(1, max_iter + 1):
        res_max: float = 0.0

        # 更新対象のインデックス：i=1..nx-2, j=0..ny-1, k=0..nz-1
        # （x=0面とx=最大面は境界条件で固定しているため，i=0とi=nx-1は除く）
        if iter_count % 2 == 1:  #   奇数回: 順方向に走査
            i_range = range(1, nx - 1)
            j_range = range(0, ny)
            k_range = range(0, nz)
        else:  #   偶数回: 逆方向に走査
            i_range = range(nx - 2, 0, -1)
            j_range = range(ny - 1, -1, -1)
            k_range = range(nz - 1, -1, -1)

        for i in i_range:
            for j in j_range:
                for k in k_range:
                    # 立方体内部はスキップ
                    if (ia < i < ib) and (ja < j < jb) and (ka < k < kb):
                        continue

                    # 隣接インデックス
                    #   m: minus, p: plus
                    im1: int = i - 1
                    ip1: int = i + 1
                    jm1: int = j - 1
                    jp1: int = j + 1
                    km1: int = k - 1
                    kp1: int = k + 1

                    # === Neumann 条件 の適用 ===
                    # 1回微分=0 を中心差分で近似
                    # 0 = dp/dx ≈ (p[i+1]-p[i-1])/(2*dx) → p[i-1]=p[i+1]

                    # ---- 外部境界 (y,z) の Neumann 条件 ----
                    if j == 0:
                        jm1 = 1
                    if j == ny - 1:
                        jp1 = ny - 2
                    if k == 0:
                        km1 = 1
                    if k == nz - 1:
                        kp1 = nz - 2

                    # ---- 立方体表面の Neumann境界条件 (x 面) ----
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
                    # 7点離散ラプラシアン
                    # 更新値を即時使用する（Gauss-Seidel 法）
                    r: float = (
                        (p[im1, j, k] - 2.0 * p[i, j, k] + p[ip1, j, k]) / dx2
                        + (p[i, jm1, k] - 2.0 * p[i, j, k] + p[i, jp1, k]) / dy2
                        + (p[i, j, km1] - 2.0 * p[i, j, k] + p[i, j, kp1]) / dz2
                    )

                    # ---- SOR 更新 ----
                    # 資料と式の形は違うが同じ意味
                    #   SOR update (Gauss–Seidel style): phi_new = (1-omega)*phi_old + omega*phi_gs
                    #   where phi_gs is the weighted neighbor average for Laplace(using latest available values).
                    p[i, j, k] = p[i, j, k] + omega * r / L
                    if abs(r) > res_max:
                        res_max = abs(r)

        # 残差の正規化（1ステップ目の res_max で割る）
        if res_max0 is None:
            res_max0 = res_max  # 初回反復の最大残差を保存

        # 最大残差の正規化
        res_max_rel: float = res_max / res_max0 if res_max0 != 0.0 else 0.0

        # 終了判定
        if res_max_rel <= tol:
            break

    # ---- 速度計算 u,v,w = grad(p) ----
    u: NDArray[np.float64] = np.zeros_like(p)  # 速度成分 (x 方向)
    v: NDArray[np.float64] = np.zeros_like(p)  # 速度成分 (y 方向)
    w: NDArray[np.float64] = np.zeros_like(p)  # 速度成分 (z 方向)

    # 境界では片側差分に切り替えるため参照点と分母を調整
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 立方体内部はスキップ
                if (ia < i < ib) and (ja < j < jb) and (ka < k < kb):
                    continue

                # 分母調整用変数
                dx1: float = 2.0 * dx
                dy1: float = 2.0 * dy
                dz1: float = 2.0 * dz

                # 隣接インデックス
                #   m: minus, p: plus
                im1: int = i - 1
                ip1: int = i + 1
                jm1: int = j - 1
                jp1: int = j + 1
                km1: int = k - 1
                kp1: int = k + 1

                # === 境界処理 ===
                # 端では片側差分にするため参照点を自分側に寄せる

                # ---- x 方向外部境界 ----
                if i == 0:
                    im1 = 0
                    dx1 = dx
                if i == nx - 1:
                    ip1 = nx - 1
                    dx1 = dx

                # 立方体 x 面（表面の外側を参照して片側差分）
                if (ja <= j <= jb) and (ka <= k <= kb):
                    if i == ib:
                        im1 = im1 + 1
                        dx1 = dx
                    if i == ia:
                        ip1 = ip1 - 1
                        dx1 = dx

                # ---- y 方向外部境界 ----
                # 端では片側差分にするため参照点を自分側に寄せる
                if j == 0:
                    jm1 = 0
                    dy1 = dy
                if j == ny - 1:
                    jp1 = ny - 1
                    dy1 = dy

                # 立方体 y 面（表面の外側を参照して片側差分）
                if (ia <= i <= ib) and (ka <= k <= kb):
                    if j == jb:
                        jm1 = jm1 + 1
                        dy1 = dy
                    if j == ja:
                        jp1 = jp1 - 1
                        dy1 = dy

                # ---- z 方向外部境界 ----
                # 端では片側差分にするため参照点を自分側に寄せる
                if k == 0:
                    km1 = 0
                    dz1 = dz
                if k == nz - 1:
                    kp1 = nz - 1
                    dz1 = dz

                # 立方体 z 面（表面の外側を参照して片側差分）
                if (ia <= i <= ib) and (ja <= j <= jb):
                    if k == kb:
                        km1 = km1 + 1
                        dz1 = dz
                    if k == ka:
                        kp1 = kp1 - 1
                        dz1 = dz

                # 勾配計算
                u[i, j, k] = (p[ip1, j, k] - p[im1, j, k]) / dx1
                v[i, j, k] = (p[i, jp1, k] - p[i, jm1, k]) / dy1
                w[i, j, k] = (p[i, j, kp1] - p[i, j, km1]) / dz1

    return p, u, v, w


def plot_potential_flow_matplotlib(
    phi: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 0.1,
    dy: float = 0.1,
    dz: float = 0.1,
    cube_bounds: Tuple[int, int, int, int, int, int] = (7, 13, 7, 13, 7, 13),
    step: int = 3,
) -> None:
    """
    ポテンシャル場のスライス図と速度ベクトル場を 3D で描画する。

    Parameters
    ----------
    phi, u, v, w : NDArray[np.float64]
        (nx, ny, nz) の配列。phi はポテンシャル，u/v/w は速度成分
    dx, dy, dz : float
        各方向の格子間隔（座標軸のスケール用）
    cube_bounds : Tuple[int, int, int, int, int, int]
        立方体（障害物）領域の範囲 (ia, ib, ja, jb, ka, kb)
    step : int
        速度ベクトル描画の間引き間隔
    """

    nx, ny, nz = phi.shape
    x = np.linspace(0, dx * (nx - 1), nx)
    y = np.linspace(0, dy * (ny - 1), ny)
    z = np.linspace(0, dz * (nz - 1), nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    def draw_cube(ax: plt.Axes) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        ia, ib, ja, jb, ka, kb = cube_bounds
        x0, x1 = x[ia], x[ib]
        y0, y1 = y[ja], y[jb]
        z0, z1 = z[ka], z[kb]
        faces = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        cube = Poly3DCollection(
            faces, facecolor="gray", edgecolor="gray", linewidths=0.8, alpha=1.0
        )
        cube.set_zorder(10)
        cube.set_sort_zpos(1.0e9)
        ax.add_collection3d(cube)

    # --- 3D slice visualization of phi ---
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.computed_zorder = False
    midx, midy, midz = nx // 2, ny // 2, nz // 2
    # insert three orthogonal filled contours as slices
    ax1.contourf(
        X[:, :, midz],
        Y[:, :, midz],
        phi[:, :, midz],
        zdir="z",
        offset=z[midz],
        levels=20,
        zorder=1,
    )
    ax1.contourf(
        X[:, midy, :],
        Z[:, midy, :],
        phi[:, midy, :],
        zdir="y",
        offset=y[midy],
        levels=20,
        zorder=1,
    )
    ax1.contourf(
        Y[midx, :, :],
        Z[midx, :, :],
        phi[midx, :, :],
        zdir="x",
        offset=x[midx],
        levels=20,
        zorder=1,
    )
    draw_cube(ax1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("3D Slices of Potential (phi)")
    plt.show()

    # --- 3D quiver of velocities (subsampled & excluding cube interior) ---
    ii = np.arange(0, nx, step)
    jj = np.arange(0, ny, step)
    kk = np.arange(0, nz, step)
    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing="ij")

    ia, ib, ja, jb, ka, kb = cube_bounds
    mask = ~((ia < II) & (II < ib) & (ja < JJ) & (JJ < jb) & (ka < KK) & (KK < kb))

    Xq = X[II, JJ, KK][mask]
    Yq = Y[II, JJ, KK][mask]
    Zq = Z[II, JJ, KK][mask]
    Uq = u[II, JJ, KK][mask]
    Vq = v[II, JJ, KK][mask]
    Wq = w[II, JJ, KK][mask]
    speed = np.sqrt(Uq**2 + Vq**2 + Wq**2)

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection="3d")
    ax2.computed_zorder = False
    cmap = plt.get_cmap("viridis")
    vmin = float(np.percentile(speed, 5))
    vmax = float(np.percentile(speed, 95))
    if vmax <= vmin:
        vmin = float(np.min(speed))
        vmax = float(np.max(speed))
    norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    colors = cmap(norm(speed))
    ax2.quiver(
        Xq,
        Yq,
        Zq,
        Uq,
        Vq,
        Wq,
        length=0.1,  # type: ignore[arg-type]
        normalize=True,
        color=colors,
        zorder=1,
    )
    draw_cube(ax2)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(speed)
    fig2.colorbar(mappable, ax=ax2, shrink=0.7, pad=0.1, label="|v|")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("3D Velocity Field (quiver)")
    plt.show()


def plot_potential_flow_pyvista(
    phi: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 0.1,
    dy: float = 0.1,
    dz: float = 0.1,
    cube_bounds: Tuple[int, int, int, int, int, int] = (7, 13, 7, 13, 7, 13),
    step: int = 3,
    cube_opacity: float = 1.0,
    arrow_scale: float = 0.1,
) -> None:
    """
    pyvista でポテンシャルのスライスと速度ベクトルを 3D 表示する。

    Parameters
    ----------
    phi, u, v, w : NDArray[np.float64]
        (nx, ny, nz) の配列。phi はポテンシャル，u/v/w は速度成分
    dx, dy, dz : float
        各方向の格子間隔（座標軸のスケール用）
    cube_bounds : Tuple[int, int, int, int, int, int]
        立方体（障害物）領域の範囲 (ia, ib, ja, jb, ka, kb)
    step : int
        速度ベクトル描画の間引き間隔
    cube_opacity : float
        ボックスの不透明度（0.0-1.0）
    arrow_scale : float
        矢印のスケール係数
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyvista が必要です: pip install pyvista") from exc

    nx, ny, nz = phi.shape
    x = np.linspace(0, dx * (nx - 1), nx)
    y = np.linspace(0, dy * (ny - 1), ny)
    z = np.linspace(0, dz * (nz - 1), nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    grid = pv.StructuredGrid(X, Y, Z)
    grid["phi"] = phi.ravel(order="F")
    velocity = np.stack((u, v, w), axis=-1)
    grid["velocity"] = velocity.reshape(-1, 3, order="F")

    midx, midy, midz = nx // 2, ny // 2, nz // 2
    slices = grid.slice_orthogonal(x=x[midx], y=y[midy], z=z[midz])

    plotter = pv.Plotter()
    plotter.add_mesh(slices, scalars="phi", cmap="viridis")

    ia, ib, ja, jb, ka, kb = cube_bounds
    bounds = (x[ia], x[ib], y[ja], y[jb], z[ka], z[kb])
    cube = pv.Box(bounds)
    plotter.add_mesh(cube, color="gray", opacity=cube_opacity)

    ii = np.arange(0, nx, step)
    jj = np.arange(0, ny, step)
    kk = np.arange(0, nz, step)
    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing="ij")

    mask = ~((ia < II) & (II < ib) & (ja < JJ) & (JJ < jb) & (ka < KK) & (KK < kb))

    Xq = x[II][mask]
    Yq = y[JJ][mask]
    Zq = z[KK][mask]
    Uq = u[II, JJ, KK][mask]
    Vq = v[II, JJ, KK][mask]
    Wq = w[II, JJ, KK][mask]
    points = np.column_stack((Xq, Yq, Zq))
    vectors = np.column_stack((Uq, Vq, Wq))
    speed = np.linalg.norm(vectors, axis=1)
    # 低速域の変化が見えるように非線形スケーリング
    speed_vis = np.sqrt(speed + 1.0e-12)

    vel_cloud = pv.PolyData(points)
    vel_cloud["velocity"] = vectors
    vel_cloud["speed"] = speed
    vel_cloud["speed_vis"] = speed_vis
    glyphs = vel_cloud.glyph(orient="velocity", scale="velocity", factor=arrow_scale)
    plotter.add_mesh(glyphs, scalars="speed_vis", cmap="viridis")

    plotter.show()


def plot_potential_flow_plotly(
    phi: NDArray[np.float64],
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 0.1,
    dy: float = 0.1,
    dz: float = 0.1,
    cube_bounds: Tuple[int, int, int, int, int, int] = (7, 13, 7, 13, 7, 13),
    step: int = 3,
    cube_opacity: float = 1.0,
    cone_sizeref: float = 0.5,
) -> None:
    """
    plotly でポテンシャルのスライスと速度ベクトルを 3D 表示する。

    Parameters
    ----------
    phi, u, v, w : NDArray[np.float64]
        (nx, ny, nz) の配列。phi はポテンシャル，u/v/w は速度成分
    dx, dy, dz : float
        各方向の格子間隔（座標軸のスケール用）
    cube_bounds : Tuple[int, int, int, int, int, int]
        立方体（障害物）領域の範囲 (ia, ib, ja, jb, ka, kb)
    step : int
        速度ベクトル描画の間引き間隔
    cube_opacity : float
        ボックスの不透明度（0.0-1.0）
    cone_sizeref : float
        コーン（矢印）のサイズ基準値
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plotly が必要です: pip install plotly") from exc

    nx, ny, nz = phi.shape
    x = np.linspace(0, dx * (nx - 1), nx)
    y = np.linspace(0, dy * (ny - 1), ny)
    z = np.linspace(0, dz * (nz - 1), nz)

    midx, midy, midz = nx // 2, ny // 2, nz // 2

    Xxy, Yxy = np.meshgrid(x, y, indexing="ij")
    Zxy = np.full_like(Xxy, z[midz])
    Xxz, Zxz = np.meshgrid(x, z, indexing="ij")
    Yxz = np.full_like(Xxz, y[midy])
    Yyz, Zyz = np.meshgrid(y, z, indexing="ij")
    Xyz = np.full_like(Yyz, x[midx])

    slice_xy = go.Surface(
        x=Xxy,
        y=Yxy,
        z=Zxy,
        surfacecolor=phi[:, :, midz],
        colorscale="Viridis",
        showscale=False,
    )
    slice_xz = go.Surface(
        x=Xxz,
        y=Yxz,
        z=Zxz,
        surfacecolor=phi[:, midy, :],
        colorscale="Viridis",
        showscale=False,
    )
    slice_yz = go.Surface(
        x=Xyz,
        y=Yyz,
        z=Zyz,
        surfacecolor=phi[midx, :, :],
        colorscale="Viridis",
        colorbar=dict(title="phi"),
    )

    ia, ib, ja, jb, ka, kb = cube_bounds
    x0, x1 = x[ia], x[ib]
    y0, y1 = y[ja], y[jb]
    z0, z1 = z[ka], z[kb]
    cube_vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    )
    cube = go.Mesh3d(
        x=cube_vertices[:, 0],
        y=cube_vertices[:, 1],
        z=cube_vertices[:, 2],
        # 12 triangles for 6 faces
        i=[0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1],
        j=[1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6],
        k=[2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5],
        color="gray",
        opacity=cube_opacity,
        name="cube",
        showscale=False,
    )

    ii = np.arange(0, nx, step)
    jj = np.arange(0, ny, step)
    kk = np.arange(0, nz, step)
    II, JJ, KK = np.meshgrid(ii, jj, kk, indexing="ij")
    mask = ~((ia < II) & (II < ib) & (ja < JJ) & (JJ < jb) & (ka < KK) & (KK < kb))

    Xq = x[II][mask]
    Yq = y[JJ][mask]
    Zq = z[KK][mask]
    Uq = u[II, JJ, KK][mask]
    Vq = v[II, JJ, KK][mask]
    Wq = w[II, JJ, KK][mask]

    cones = go.Cone(
        x=Xq,
        y=Yq,
        z=Zq,
        u=Uq,
        v=Vq,
        w=Wq,
        sizemode="absolute",
        sizeref=cone_sizeref,
        anchor="tail",
        colorscale="Viridis",
        showscale=False,
        name="velocity",
    )

    fig = go.Figure(data=[slice_xy, slice_xz, slice_yz, cube, cones])
    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        title="Potential and Velocity Field",
    )
    fig.show()


if __name__ == "__main__":
    phi, u, v, w = solve_potential_flow_cube()
    print("phi shape:", phi.shape)
    print(
        "max |u|,|v|,|w|:",
        float(np.max(np.abs(u))),
        float(np.max(np.abs(v))),
        float(np.max(np.abs(w))),
    )

    # plot_potential_flow_matplotlib(phi, u, v, w)
    # plot_potential_flow_pyvista(phi, u, v, w, arrow_scale=0.3)
    plot_potential_flow_plotly(phi, u, v, w, cone_sizeref=0.3)
