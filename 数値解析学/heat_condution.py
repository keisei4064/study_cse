# Generate and save 3D visualizations again with robust plotting (slices + quiver)
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple


def solve_potential_flow_cube(
    nx: int = 21,
    ny: int = 21,
    nz: int = 21,
    dx: float = 0.1,
    dy: float = 0.1,
    dz: float = 0.1,
    omega: float = 1.5,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    p: NDArray[np.float64] = np.zeros((nx, ny, nz), dtype=np.float64)
    u: NDArray[np.float64] = np.zeros_like(p)
    v: NDArray[np.float64] = np.zeros_like(p)
    w: NDArray[np.float64] = np.zeros_like(p)

    dx2: float = dx * dx
    dy2: float = dy * dy
    dz2: float = dz * dz
    c0: float = 2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2

    ia, ib = 7, 13
    ja, jb = 7, 13
    ka, kb = 7, 13

    p[0, :, :] = 0.0
    p[nx - 1, :, :] = 1.0

    rmax1: float | None = None

    for na in range(1, max_iter + 1):
        rmax: float = 0.0

        if na % 2 == 1:
            i_indices = range(1, nx - 1)
            j_indices = range(0, ny)
            k_indices = range(0, nz)
        else:
            i_indices = range(nx - 2, 0, -1)
            j_indices = range(ny - 1, -1, -1)
            k_indices = range(nz - 1, -1, -1)

        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    if (ia < i < ib) and (ja < j < jb) and (ka < k < kb):
                        continue

                    im1: int = i - 1
                    ip1: int = i + 1
                    jm1: int = j - 1
                    jp1: int = j + 1
                    km1: int = k - 1
                    kp1: int = k + 1

                    if j == 0:
                        jm1 = 1
                    if j == ny - 1:
                        jp1 = ny - 2
                    if k == 0:
                        km1 = 1
                    if k == nz - 1:
                        kp1 = nz - 2

                    if (ja <= j <= jb) and (ka <= k <= kb):
                        if i == ib:
                            im1 = im1 + 2
                        if i == ia:
                            ip1 = ip1 - 2

                    if (ia <= i <= ib) and (ka <= k <= kb):
                        if j == jb:
                            jm1 = jm1 + 2
                        if j == ja:
                            jp1 = jp1 - 2

                    if (ia <= i <= ib) and (ja <= j <= jb):
                        if k == kb:
                            km1 = km1 + 2
                        if k == ka:
                            kp1 = kp1 - 2

                    r: float = (
                        (p[im1, j, k] - 2.0 * p[i, j, k] + p[ip1, j, k]) / (dx * dx)
                        + (p[i, jm1, k] - 2.0 * p[i, j, k] + p[i, jp1, k]) / (dy * dy)
                        + (p[i, j, km1] - 2.0 * p[i, j, k] + p[i, j, kp1]) / (dz * dz)
                    )
                    p[i, j, k] = p[i, j, k] + 1.5 * r / c0
                    rmax = max(rmax, abs(r))

        if rmax1 is None:
            rmax1 = rmax
        rnorm: float = rmax / rmax1 if rmax1 != 0.0 else 0.0
        if rnorm <= tol:
            break

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
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

                if i == 0:
                    im1 = 0
                    dx1 = dx
                if i == nx - 1:
                    ip1 = nx - 1
                    dx1 = dx

                if (ja <= j <= jb) and (ka <= k <= kb):
                    if i == ib:
                        im1 = im1 + 1
                        dx1 = dx
                    if i == ia:
                        ip1 = ip1 - 1
                        dx1 = dx

                if j == 0:
                    jm1 = 0
                    dy1 = dy
                if j == ny - 1:
                    jp1 = ny - 1
                    dy1 = dy

                if (ia <= i <= ib) and (ka <= k <= kb):
                    if j == jb:
                        jm1 = jm1 + 1
                        dy1 = dy
                    if j == ja:
                        jp1 = jp1 - 1
                        dy1 = dy

                if k == 0:
                    km1 = 0
                    dz1 = dz
                if k == nz - 1:
                    kp1 = nz - 1
                    dz1 = dz

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


phi, u, v, w = solve_potential_flow_cube()

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
fig1.savefig("phi_slices_3d.png", dpi=220, bbox_inches="tight")
plt.show()
# plt.close(fig1)

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
fig2.savefig("velocity_quiver.png", dpi=220, bbox_inches="tight")
plt.show()
# plt.close(fig2)
