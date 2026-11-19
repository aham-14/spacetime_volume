"""
engine/sdf_shapes.py — Signed Distance Field (SDF) primitives library.

WHY SDFs FOR 3D PRINTING:
--------------------------
Hard binary masks create discontinuous (0/1) voxel grids. When marching_cubes
extracts an isosurface at level=0.5, it hits the step edge exactly, producing
a staircase surface on every face.

SDFs store the SIGNED distance from each pixel to the nearest surface boundary:
  - Negative  → inside the shape
  - Zero      → exactly on the boundary
  - Positive  → outside the shape

Converting via a soft sigmoid gives a smooth gradient around the boundary:
   intensity(d) = σ(−d / edge_width) ∈ (0, 1)

Marching cubes then has a smooth scalar field to interpolate, producing
vertex positions with sub-pixel accuracy and no staircasing.

All functions return float32 SDFs in pixel units.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sdf_disc",
    "sdf_rect",
    "sdf_rounded_rect",
    "sdf_ring",
    "sdf_capsule",
    "sdf_union",
    "sdf_intersect",
    "sdf_subtract",
    "sdf_smooth_union",
    "sdf_to_intensity",
    "make_grids",
]


def make_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, X) coordinate grids, shape (H, W) each — broadcastable."""
    return np.mgrid[0:h, 0:w]


def sdf_disc(h: int, w: int, cy: float, cx: float, r: float) -> np.ndarray:
    """SDF of a filled disc: d < 0 inside, d > 0 outside."""
    ys, xs = np.mgrid[0:h, 0:w]
    return (np.hypot(ys - cy, xs - cx) - r).astype(np.float32)


def sdf_rect(h: int, w: int, cy: float, cx: float, hh: float, hw: float) -> np.ndarray:
    """Exact SDF of an axis-aligned rectangle (box)."""
    ys, xs = np.mgrid[0:h, 0:w]
    dy = np.abs(ys - cy) - hh
    dx = np.abs(xs - cx) - hw
    outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2)
    inside  = np.minimum(np.maximum(dx, dy), 0.0)
    return (outside + inside).astype(np.float32)


def sdf_rounded_rect(
    h: int,
    w: int,
    cy: float,
    cx: float,
    hh: float,
    hw: float,
    r: float = 8.0,
) -> np.ndarray:
    """Exact Euclidean Signed Distance Field (SDF) of a rounded rectangle.

    h, w : grid dimensions
    cy, cx : center coordinates
    hh, hw : half-height and half-width of the total outer bounding box
    r : corner radius (clamped to [0, min(hh, hw)])
    """
    r_clamped = float(np.clip(r, 0.0, min(hh, hw)))
    ys, xs = np.mgrid[0:h, 0:w]
    qy = np.abs(ys - cy) - hh + r_clamped
    qx = np.abs(xs - cx) - hw + r_clamped
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return (outside + inside - r_clamped).astype(np.float32)


def sdf_ring(h: int, w: int, cy: float, cx: float, r_outer: float, r_inner: float) -> np.ndarray:
    """SDF of an annulus (ring) = disc_outer - disc_inner."""
    r_mid   = (r_outer + r_inner) / 2.0
    half_th = (r_outer - r_inner) / 2.0
    ys, xs = np.mgrid[0:h, 0:w]
    return (np.abs(np.hypot(ys - cy, xs - cx) - r_mid) - half_th).astype(np.float32)


def sdf_capsule(
    h: int, w: int,
    ay: float, ax: float,
    by: float, bx: float,
    r: float,
) -> np.ndarray:
    """SDF of a capsule (stadium): endpoints A, B and radius r."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dy_ab, dx_ab = by - ay, bx - ax
    len2 = dy_ab ** 2 + dx_ab ** 2 + 1e-12
    t = np.clip(((ys - ay) * dy_ab + (xs - ax) * dx_ab) / len2, 0.0, 1.0)
    proj_y = ay + t * dy_ab
    proj_x = ax + t * dx_ab
    return (np.hypot(ys - proj_y, xs - proj_x) - r).astype(np.float32)


def sdf_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)


def sdf_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def sdf_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, -b)


def sdf_smooth_union(a: np.ndarray, b: np.ndarray, k: float = 8.0) -> np.ndarray:
    """Smooth-minimum union (Quilez, 2013) with blend radius k."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return np.where(
        np.abs(a - b) < k,
        a * h + b * (1.0 - h) - k * h * (1.0 - h),
        np.minimum(a, b),
    ).astype(np.float32)


def sdf_to_intensity(
    sdf: np.ndarray,
    edge_width: float = 2.0,
    invert: bool = False,
) -> np.ndarray:
    """Map signed-distance field to smooth float32 intensity in [0, 1]."""
    sign = 1.0 if not invert else -1.0
    return (1.0 / (1.0 + np.exp(sign * sdf / edge_width))).astype(np.float32)

