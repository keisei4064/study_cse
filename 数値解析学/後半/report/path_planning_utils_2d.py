from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


def _closest_index(values: FloatArray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def goal_disk_indices(
    goal: Tuple[float, float],
    radius: float,
    xs: FloatArray,
    ys: FloatArray,
) -> list[Tuple[int, int]]:
    nx, ny = xs.size, ys.size
    gx, gy = goal
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    r_ix = int(np.ceil(radius / dx))
    r_iy = int(np.ceil(radius / dy))

    ig = _closest_index(xs, gx)
    jg = _closest_index(ys, gy)
    indices: list[Tuple[int, int]] = []
    for i in range(max(0, ig - r_ix), min(nx, ig + r_ix + 1)):
        for j in range(max(0, jg - r_iy), min(ny, jg + r_iy + 1)):
            if (xs[i] - gx) ** 2 + (ys[j] - gy) ** 2 <= radius * radius:
                indices.append((i, j))
    return indices
