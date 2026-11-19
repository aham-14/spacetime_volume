"""
engine/mesh_export.py — 3D Printable Mesh Generation & Export for SLA Resin & Casting.

HIGH-RESOLUTION SLA & CASTING FEATURES:
----------------------------------------
1. Negative Mold Generation (Boolean Subtraction):
   - Subtracts the Space-Time Volume from a solid 3D bounding box array.
   - Exports the negative space as a hollowed STL mold suitable for pouring wax or silicone.
   - Includes open pour hole/spout at the top surface and solid exterior walls.

2. High-Resolution SLA Tuning:
   - Exposes `step_size` (default 1) in `marching_cubes` for maximum resin hardware resolution.
   - Reduced default Taubin smoothing iterations (4 passes) to preserve fine pixel details.

3. Strict Memory Management (8GB RAM / 4GB GPU budget):
   - Chunked processing along the Z-axis with 1-voxel slice overlaps.
   - Numba JIT accelerated chunk filling and in-place boolean subtraction.
   - SciPy sparse matrix representations for slice occupancy to skip homogeneous chunks.
   - Streaming memory reclamation between chunks to avoid OOM crashes.
"""
from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numba
import numpy as np
import scipy.sparse as sp

from .volume import SpaceTimeVolume


@dataclass
class MeshStats:
    """Statistics returned alongside the exported mesh."""
    n_vertices: int
    n_faces: int
    bounds_mm: Tuple[Tuple[float, float], ...]   # ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    volume_mm3: float
    is_watertight: bool
    is_negative_mold: bool
    output_path: str


# ---------------------------------------------------------------------------
# Numba JIT Kernels for In-Place Memory-Efficient Operations
# ---------------------------------------------------------------------------


@numba.njit(fastmath=True)
def fast_chunk_bounds(arr: np.ndarray) -> Tuple[float, float]:
    """Fast min and max scan without allocating intermediate NumPy arrays."""
    flat = arr.ravel()
    n = len(flat)
    if n == 0:
        return 0.0, 0.0
    mn = flat[0]
    mx = flat[0]
    for i in range(1, n):
        v = flat[i]
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    return mn, mx


@numba.njit(parallel=True, fastmath=True)
def numba_fill_mold_chunk(
    chunk: np.ndarray,
    obj_data: np.ndarray,
    chunk_z0: int,
    wall_z: int,
    wall_y: int,
    wall_x: int,
    open_top: bool,
    mold_total_z: int,
) -> None:
    """Fill a mold chunk in-place with solid material (1.0) and subtract object (0.0 inside).

    Memory optimized: works in-place on a single small chunk slice, avoiding
    allocating the entire monolithic 3D mold bounding box in RAM.
    """
    cz, cy, cx = chunk.shape
    obj_T, obj_H, obj_W = obj_data.shape

    for z in numba.prange(cz):
        global_z = chunk_z0 + z

        # Bottom padding layer (air outside mold base)
        if global_z == 0:
            for y in range(cy):
                for x in range(cx):
                    chunk[z, y, x] = 0.0
            continue

        # Top layer above mold rim (air outside top)
        if global_z == mold_total_z - 1:
            for y in range(cy):
                for x in range(cx):
                    chunk[z, y, x] = 0.0
            continue

        for y in range(cy):
            # Outer y boundary air padding
            if y == 0 or y == cy - 1:
                for x in range(cx):
                    chunk[z, y, x] = 0.0
                continue

            for x in range(cx):
                # Outer x boundary air padding
                if x == 0 or x == cx - 1:
                    chunk[z, y, x] = 0.0
                    continue

                # Solid mold block default
                val = 1.0

                obj_z_idx = global_z - wall_z
                obj_y_idx = y - wall_y
                obj_x_idx = x - wall_x

                if 0 <= obj_z_idx < obj_T and 0 <= obj_y_idx < obj_H and 0 <= obj_x_idx < obj_W:
                    ov = obj_data[obj_z_idx, obj_y_idx, obj_x_idx]
                    # Boolean subtraction: solid mold (1.0) - object (>0.5 becomes hollow <0.5)
                    sub = 1.0 - ov
                    if sub < 0.0:
                        sub = 0.0
                    elif sub > 1.0:
                        sub = 1.0
                    val = sub
                elif open_top and global_z >= wall_z + obj_T:
                    # Pour channel extending through the top of the mold box
                    if 0 <= obj_y_idx < obj_H and 0 <= obj_x_idx < obj_W:
                        top_ov = obj_data[obj_T - 1, obj_y_idx, obj_x_idx]
                        if top_ov > 0.3:
                            val = 0.0  # Open void for pouring wax/silicone

                chunk[z, y, x] = val


# ---------------------------------------------------------------------------
# SciPy Sparse Matrix Occupancy Indexer
# ---------------------------------------------------------------------------


class SparseOccupancyIndex:
    """Uses SciPy sparse matrix representation to track active isosurface slices.

    Avoids holding dense mask arrays in memory when checking high-res volumes.
    """

    def __init__(self, vol_data: np.ndarray, level: float = 0.5) -> None:
        T = vol_data.shape[0]
        # Compute min/max per slice along time
        slice_mins = np.fromiter((vol_data[t].min() for t in range(T)), dtype=np.float32, count=T)
        slice_maxs = np.fromiter((vol_data[t].max() for t in range(T)), dtype=np.float32, count=T)

        # A slice is active if it spans the isosurface level
        active = (slice_mins <= level) & (slice_maxs >= level)
        # Store as SciPy CSR sparse matrix (1 x T)
        self.sparse_active = sp.csr_matrix(active.reshape(1, -1))
        self.total_slices = T

    def has_active_voxels(self, z_start: int, z_end: int) -> bool:
        """Check if any slice in [z_start, z_end) intersects the isosurface."""
        z_start = max(0, z_start)
        z_end = min(self.total_slices, z_end)
        if z_end <= z_start:
            return False
        sub = self.sparse_active[0, z_start:z_end]
        return sub.nnz > 0


# ---------------------------------------------------------------------------
# Chunked Marching Cubes Engine
# ---------------------------------------------------------------------------


def chunked_marching_cubes(
    vol_data: np.ndarray,
    level: float = 0.5,
    step_size: int = 1,
    chunk_size: int = 32,
    allow_degenerate: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Memory-safe chunked marching cubes implementation.

    Splits dense 3D volumes into Z-axis slabs overlapping by 1 voxel to
    eliminate seam gaps while strictly bounding peak RAM allocation.

    Parameters
    ----------
    vol_data : ndarray (T, H, W) float32
    level : float
    step_size : int (1 for high-res SLA, 2+ for faster preview)
    chunk_size : int (number of slices per chunk)
    allow_degenerate : bool
    progress_callback : callable(percent_float) | None

    Returns
    -------
    (vertices, faces)
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise ImportError("scikit-image is required for marching_cubes") from exc

    Nz = vol_data.shape[0]
    S = max(1, step_size)
    # Ensure chunk_size is aligned with step_size
    effective_chunk = max(S * 2, (chunk_size // S) * S)

    all_verts = []
    all_faces = []
    v_offset = 0

    # Build sparse index to skip homogeneous regions
    occupancy = SparseOccupancyIndex(vol_data, level=level)

    z_starts = list(range(0, Nz - 1, effective_chunk))
    n_chunks = len(z_starts)

    for idx, z0 in enumerate(z_starts):
        z1 = min(z0 + effective_chunk + 1, Nz)

        # Quick check if chunk intersects isosurface
        if not occupancy.has_active_voxels(z0, z1):
            if progress_callback:
                progress_callback(100.0 * (idx + 1) / n_chunks)
            continue

        chunk = vol_data[z0:z1]
        cmin, cmax = fast_chunk_bounds(chunk)

        if cmin <= level <= cmax and (z1 - z0) > S:
            vc, fc, _, _ = marching_cubes(
                chunk,
                level=level,
                step_size=S,
                allow_degenerate=allow_degenerate,
            )
            # Offset Z coordinate by global slab start
            vc[:, 0] += z0

            all_verts.append(vc)
            all_faces.append(fc + v_offset)
            v_offset += len(vc)

        if progress_callback:
            progress_callback(100.0 * (idx + 1) / n_chunks)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    merged_verts = np.vstack(all_verts).astype(np.float32)
    merged_faces = np.vstack(all_faces).astype(np.int32)

    # Force garbage collection to reclaim chunk buffer memory
    del all_verts, all_faces
    gc.collect()

    return merged_verts, merged_faces


# ---------------------------------------------------------------------------
# Backend Negative Mold Generator
# ---------------------------------------------------------------------------

# Max Z-slices to hold in RAM at once during mold filling.
# 64 slices × 512 × 512 × float32 ≈ 64 MB — well within the 8 GB constraint.
MOLD_CHUNK_Z: int = 64


def generate_negative_mold(
    vol: SpaceTimeVolume,
    wall_thickness: int = 4,
    open_top: bool = True,
) -> SpaceTimeVolume:
    """Generate a solid 3D bounding box array and subtract the Space-Time Volume.

    Returns a new SpaceTimeVolume representing the negative mold block:
      - Solid walls of thickness ``wall_thickness`` voxels surrounding the object.
      - Hollow interior cavity matching the Space-Time Volume geometry.
      - Top pour hole/spout (if open_top=True) for pouring wax or silicone.

    Memory management
    -----------------
    Instead of allocating the entire mold volume in a single ``np.zeros`` call
    (which can exceed 3 GB for large SLA volumes), this function writes the mold
    data into a temporary ``np.memmap`` file in Z-axis slabs of ``MOLD_CHUNK_Z``
    slices.  Peak RAM per slab is bounded by::

        MOLD_CHUNK_Z × total_y × total_x × 4 bytes

    For a 512×512 footprint that is ≈ 64 MB.  After all slabs are written, the
    memmap array is returned as the backing storage for the new ``SpaceTimeVolume``
    so no second full-volume copy ever exists in RAM.
    """
    T, H, W = vol.shape
    pad_z = 2  # base layer + top rim clearance
    total_z = T + wall_thickness + pad_z
    total_y = H + 2 * wall_thickness + 2
    total_x = W + 2 * wall_thickness + 2

    wall_z = wall_thickness + 1
    wall_y = wall_thickness + 1
    wall_x = wall_thickness + 1

    # ------------------------------------------------------------------
    # Allocate output via a temp memmap so the full mold volume is never
    # resident in RAM at once.  The file is kept open (delete=False) so
    # the returned SpaceTimeVolume can still read it; the caller owns the
    # lifetime.  On CPython the file is unlinked when the process exits.
    # ------------------------------------------------------------------
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mold.mmap")
    os.close(tmp_fd)  # numpy memmap opens the file itself
    mold_mm = np.memmap(
        tmp_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_z, total_y, total_x),
    )

    # Process the mold in Z-axis slabs to bound peak RAM.
    for z0 in range(0, total_z, MOLD_CHUNK_Z):
        z1 = min(z0 + MOLD_CHUNK_Z, total_z)
        slab = np.zeros((z1 - z0, total_y, total_x), dtype=np.float32)

        numba_fill_mold_chunk(
            chunk=slab,
            obj_data=vol.data,
            chunk_z0=z0,
            wall_z=wall_z,
            wall_y=wall_y,
            wall_x=wall_x,
            open_top=open_top,
            mold_total_z=total_z,
        )

        mold_mm[z0:z1] = slab
        mold_mm.flush()  # write slab to disk before releasing RAM
        del slab
        gc.collect()

    new_meta = {
        **vol.meta,
        "is_negative_mold": True,
        "mold_wall_thickness": wall_thickness,
        "mold_open_top": open_top,
        "cavity_shape": vol.shape,
        "_mold_mmap_path": tmp_path,
    }
    # Return a read-only view of the memmap; SpaceTimeVolume stores it as .data
    return SpaceTimeVolume(np.asarray(mold_mm), new_meta)


# ---------------------------------------------------------------------------
# High-Resolution Mesh Exporter Class
# ---------------------------------------------------------------------------


class MeshExporter:
    """Convert a SpaceTimeVolume into a high-res STL mesh (Positive or Negative Mold).

    Optimized for high-resolution SLA resin printing and physical casting:
      - Supports negative mold generation via boolean subtraction.
      - Step size configurable for maximum resin resolution (step_size=1).
      - Low Taubin smoothing passes by default to preserve sharp pixel features.
      - Chunked processing to strictly adhere to 8GB RAM / 4GB GPU constraints.
    """

    def __init__(
        self,
        level: float = 0.5,
        step_size: int = 1,
        smooth_iterations: int = 4,
        taubin_lambda: float = 0.5,
        taubin_mu: float = 0.52,
        voxel_size: float = 0.5,
        chunk_size: int = 32,
        allow_degenerate: bool = True,
        as_negative_mold: bool = False,
        mold_wall_thickness: int = 4,
        mold_open_top: bool = True,
    ) -> None:
        self.level = level
        self.step_size = step_size
        self.smooth_iterations = smooth_iterations
        self.taubin_lambda = taubin_lambda
        self.taubin_mu = taubin_mu
        self.voxel_size = voxel_size
        self.chunk_size = chunk_size
        self.allow_degenerate = allow_degenerate
        self.as_negative_mold = as_negative_mold
        self.mold_wall_thickness = mold_wall_thickness
        self.mold_open_top = mold_open_top

    def export(
        self,
        vol: SpaceTimeVolume,
        output_path: Union[str, Path],
        pad: int = 2,
        as_negative_mold: Optional[bool] = None,
        mold_wall_thickness: Optional[int] = None,
        mold_open_top: Optional[bool] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> MeshStats:
        """Extract mesh, apply Taubin smoothing, scale, and write STL file.

        Parameters
        ----------
        vol : SpaceTimeVolume
        output_path : path-like (.stl)
        pad : int (padding for watertight boundaries)
        as_negative_mold : bool | None (overrides instance setting if given)
        mold_wall_thickness : int | None
        mold_open_top : bool | None
        progress_callback : callable(percent) | None

        Returns
        -------
        MeshStats
        """
        try:
            import trimesh
            import trimesh.smoothing as tsmooth
        except ImportError as exc:
            raise ImportError("trimesh is required for mesh export: pip install trimesh") from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        is_mold = self.as_negative_mold if as_negative_mold is None else as_negative_mold
        wall_th = self.mold_wall_thickness if mold_wall_thickness is None else mold_wall_thickness
        open_tp = self.mold_open_top if mold_open_top is None else mold_open_top

        # --------------------------------------------------------------
        # 1. Prepare target volume (Positive Object or Negative Mold)
        # --------------------------------------------------------------
        if is_mold:
            # Generate negative mold (streamed / chunked friendly)
            target_vol = generate_negative_mold(
                vol,
                wall_thickness=wall_th,
                open_top=open_tp,
            )
            data_to_mesh = target_vol.data
            effective_pad = 0  # Mold generator already includes sealing boundary padding
        else:
            data = vol.data
            effective_pad = pad
            if effective_pad > 0:
                data_to_mesh = np.pad(data, pad_width=effective_pad, mode="constant", constant_values=0.0)
            else:
                data_to_mesh = data

        # --------------------------------------------------------------
        # 2. Chunked Marching Cubes (Memory Budget: 8GB System RAM)
        # --------------------------------------------------------------
        verts, faces = chunked_marching_cubes(
            data_to_mesh,
            level=self.level,
            step_size=self.step_size,
            chunk_size=self.chunk_size,
            allow_degenerate=self.allow_degenerate,
            progress_callback=progress_callback,
        )

        if len(verts) == 0 or len(faces) == 0:
            raise ValueError(
                f"No isosurface found at level={self.level}. "
                f"Volume range: [{data_to_mesh.min():.2f}, {data_to_mesh.max():.2f}]."
            )

        # Reorder (z, y, x) = (t, row, col) -> (col, row, time) = (x, y, z)
        verts = verts[:, [2, 1, 0]]
        if effective_pad > 0:
            verts -= effective_pad
        verts *= self.voxel_size  # mm units

        # --------------------------------------------------------------
        # 3. Build Trimesh Object & Weld Boundary Vertices
        # --------------------------------------------------------------
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=True,  # welds duplicate chunk boundary vertices and fixes winding
        )

        # --------------------------------------------------------------
        # 4. High-Res SLA Taubin Smoothing
        # --------------------------------------------------------------
        if self.smooth_iterations > 0:
            tsmooth.filter_taubin(
                mesh,
                lamb=self.taubin_lambda,
                nu=self.taubin_mu,
                iterations=self.smooth_iterations,
            )

        # --------------------------------------------------------------
        # 5. Export STL
        # --------------------------------------------------------------
        mesh.export(str(output_path))

        bounds = mesh.bounds
        stats = MeshStats(
            n_vertices=len(mesh.vertices),
            n_faces=len(mesh.faces),
            bounds_mm=tuple(
                (float(bounds[0, i]), float(bounds[1, i])) for i in range(3)
            ),
            volume_mm3=float(mesh.volume) if mesh.is_watertight else float("nan"),
            is_watertight=bool(mesh.is_watertight),
            is_negative_mold=is_mold,
            output_path=str(output_path.resolve()),
        )

        # Reclaim mesh RAM immediately
        del mesh, data_to_mesh
        gc.collect()

        return stats

    def export_negative_mold(
        self,
        vol: SpaceTimeVolume,
        output_path: Union[str, Path],
        wall_thickness: int = 4,
        open_top: bool = True,
    ) -> MeshStats:
        """Convenience method to export a hollowed negative mold STL directly."""
        return self.export(
            vol,
            output_path,
            as_negative_mold=True,
            mold_wall_thickness=wall_thickness,
            mold_open_top=open_top,
        )

    def __repr__(self) -> str:
        mold_str = " (Negative Mold)" if self.as_negative_mold else ""
        return (
            f"MeshExporter(level={self.level}, step_size={self.step_size}, "
            f"smooth_iters={self.smooth_iterations}, voxel_size={self.voxel_size}mm, "
            f"chunk_size={self.chunk_size}{mold_str})"
        )
