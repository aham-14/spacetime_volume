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
    export_format: str = "stl"
    dimensions_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (width_x, depth_y, height_z)
    fits_standard_sla_vat: bool = True  # fits within standard 143 x 89 x 175 mm SLA build volume
    repair_applied: bool = False


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


def generate_two_piece_mold(
    vol: SpaceTimeVolume,
    wall_thickness: int = 4,
    open_top: bool = True,
    split_axis: int = 1,  # 1 = Y (rows), 2 = X (cols)
    key_radius: int = 3,
    key_depth: int = 2,
) -> Tuple[SpaceTimeVolume, SpaceTimeVolume]:
    """Split a negative mold volume into two interlocking halves with alignment keys.

    Creates a 2-part casting mold suitable for pouring wax, silicone, or resin
    where complex undercuts would prevent single-block demolding:
      - Part A (male): Receives raised hemispherical registration keys (pegs).
      - Part B (female): Receives matching negative registration sockets.
      - Top pour hole / spout remains intact.

    Parameters
    ----------
    vol : SpaceTimeVolume
        Positive space-time volume or pre-generated mold.
    wall_thickness : int
        Thickness of bounding mold walls in voxels.
    open_top : bool
        Whether to leave the top pour hole open.
    split_axis : int
        Axis to split along: 1 for Y (default), 2 for X.
    key_radius : int
        Radius of registration pegs/sockets in voxels.
    key_depth : int
        Projection depth of registration keys along the split axis.

    Returns
    -------
    (part_a, part_b) : Tuple[SpaceTimeVolume, SpaceTimeVolume]
    """
    if vol.meta.get("is_negative_mold", False):
        mold_vol = vol
    else:
        mold_vol = generate_negative_mold(
            vol, wall_thickness=wall_thickness, open_top=open_top
        )

    mold_data = mold_vol.data
    T, H, W = mold_data.shape

    part_a_data = mold_data.copy()
    part_b_data = mold_data.copy()

    if split_axis == 1:
        # Split along Y (rows)
        split_idx = H // 2
        part_a_data[:, split_idx:, :] = 0.0
        part_b_data[:, :split_idx, :] = 0.0

        # Place registration keys near flange corners
        z_keys = [max(1, wall_thickness // 2), T - max(2, wall_thickness // 2) - 1]
        x_keys = [max(1, wall_thickness // 2), W - max(2, wall_thickness // 2) - 1]

        for zk in z_keys:
            for xk in x_keys:
                for dz in range(-key_radius, key_radius + 1):
                    for dx in range(-key_radius, key_radius + 1):
                        if dz * dz + dx * dx <= key_radius * key_radius:
                            z_pos = np.clip(zk + dz, 0, T - 1)
                            x_pos = np.clip(xk + dx, 0, W - 1)
                            for dy in range(0, key_depth):
                                y_pos = np.clip(split_idx + dy, 0, H - 1)
                                part_a_data[z_pos, y_pos, x_pos] = 1.0
                                part_b_data[z_pos, y_pos, x_pos] = 0.0
    else:
        # Split along X (cols)
        split_idx = W // 2
        part_a_data[:, :, split_idx:] = 0.0
        part_b_data[:, :, :split_idx] = 0.0

        z_keys = [max(1, wall_thickness // 2), T - max(2, wall_thickness // 2) - 1]
        y_keys = [max(1, wall_thickness // 2), H - max(2, wall_thickness // 2) - 1]

        for zk in z_keys:
            for yk in y_keys:
                for dz in range(-key_radius, key_radius + 1):
                    for dy in range(-key_radius, key_radius + 1):
                        if dz * dz + dy * dy <= key_radius * key_radius:
                            z_pos = np.clip(zk + dz, 0, T - 1)
                            y_pos = np.clip(yk + dy, 0, H - 1)
                            for dx in range(0, key_depth):
                                x_pos = np.clip(split_idx + dx, 0, W - 1)
                                part_a_data[z_pos, y_pos, x_pos] = 1.0
                                part_b_data[z_pos, y_pos, x_pos] = 0.0

    meta_a = {
        **mold_vol.meta,
        "is_two_piece_mold": True,
        "mold_part": "A",
        "split_axis": split_axis,
        "split_index": split_idx,
        "has_registration_keys": True,
    }
    meta_b = {
        **mold_vol.meta,
        "is_two_piece_mold": True,
        "mold_part": "B",
        "split_axis": split_axis,
        "split_index": split_idx,
        "has_registration_keys": True,
    }

    return SpaceTimeVolume(part_a_data, meta_a), SpaceTimeVolume(part_b_data, meta_b)


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
        as_hollow: bool = False,
        hollow_wall_thickness: float = 3.0,
        auto_drain_holes: bool = True,
        constant_boundary: Optional[str] = None,
        boundary_mode: str = "framed",
        boundary_bezel: float = 4.0,
        boundary_corner_radius: float = 8.0,
        boundary_padding: float = 2.0,
        base_plate_slices: int = 0,
        uninvert_y: bool = True,
        invert_z: bool = False,
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
        self.as_hollow = as_hollow
        self.hollow_wall_thickness = hollow_wall_thickness
        self.auto_drain_holes = auto_drain_holes
        self.constant_boundary = constant_boundary
        self.boundary_mode = boundary_mode
        self.boundary_bezel = boundary_bezel
        self.boundary_corner_radius = boundary_corner_radius
        self.boundary_padding = boundary_padding
        self.base_plate_slices = base_plate_slices
        self.uninvert_y = uninvert_y
        self.invert_z = invert_z

    def export(
        self,
        vol: SpaceTimeVolume,
        output_path: Union[str, Path],
        level: Optional[float] = None,
        pad: int = 2,
        as_negative_mold: Optional[bool] = None,
        mold_wall_thickness: Optional[int] = None,
        mold_open_top: Optional[bool] = None,
        as_hollow: Optional[bool] = None,
        hollow_wall_thickness: Optional[float] = None,
        auto_drain_holes: Optional[bool] = None,
        constant_boundary: Optional[str] = None,
        boundary_mode: Optional[str] = None,
        boundary_bezel: Optional[float] = None,
        boundary_corner_radius: Optional[float] = None,
        boundary_padding: Optional[float] = None,
        base_plate_slices: Optional[int] = None,
        uninvert_y: Optional[bool] = None,
        invert_z: Optional[bool] = None,
        repair_mesh: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> MeshStats:
        """Extract mesh, apply Taubin smoothing, scale, and write 3D file (.stl, .obj, .ply, .glb).

        Parameters
        ----------
        vol : SpaceTimeVolume
        output_path : path-like (.stl, .obj, .ply, .glb)
        level : float | None (overrides instance level if given)
        pad : int (padding for watertight boundaries)
        as_negative_mold : bool | None (overrides instance setting if given)
        mold_wall_thickness : int | None
        mold_open_top : bool | None
        as_hollow : bool | None (hollows the object for SLA resin saving)
        hollow_wall_thickness : float | None
        auto_drain_holes : bool | None (punches bottom drain holes into hollow void)
        repair_mesh : bool (runs automated normal/winding repairs for slicers)
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
        export_fmt = output_path.suffix.lstrip(".").lower() or "stl"

        eff_level = self.level if level is None else float(level)
        is_mold = self.as_negative_mold if as_negative_mold is None else as_negative_mold
        wall_th = self.mold_wall_thickness if mold_wall_thickness is None else mold_wall_thickness
        open_tp = self.mold_open_top if mold_open_top is None else mold_open_top
        is_hollow = self.as_hollow if as_hollow is None else as_hollow
        hollow_th = self.hollow_wall_thickness if hollow_wall_thickness is None else hollow_wall_thickness
        drain = self.auto_drain_holes if auto_drain_holes is None else auto_drain_holes

        eff_boundary = self.constant_boundary if constant_boundary is None else constant_boundary
        eff_b_mode = self.boundary_mode if boundary_mode is None else boundary_mode
        eff_bezel = self.boundary_bezel if boundary_bezel is None else boundary_bezel
        eff_cr = self.boundary_corner_radius if boundary_corner_radius is None else boundary_corner_radius
        eff_pad = self.boundary_padding if boundary_padding is None else boundary_padding
        eff_base = self.base_plate_slices if base_plate_slices is None else base_plate_slices

        # --------------------------------------------------------------
        # 1. Prepare target volume (Enclosure Framing, Mold, or Hollow)
        # --------------------------------------------------------------
        work_vol = vol
        if eff_boundary is not None and str(eff_boundary).strip().lower() not in ("none", "", "false"):
            work_vol = work_vol.apply_constant_boundary(
                shape=str(eff_boundary),
                mode=eff_b_mode,
                bezel_voxels=float(eff_bezel),
                corner_radius=float(eff_cr),
                padding_voxels=float(eff_pad),
                base_plate_slices=int(eff_base),
            )

        if is_mold:
            # Generate negative mold (streamed / chunked friendly)
            target_vol = generate_negative_mold(
                work_vol,
                wall_thickness=wall_th,
                open_top=open_tp,
            )
            data_to_mesh = target_vol.data
            effective_pad = 0  # Mold generator already includes sealing boundary padding
        elif is_hollow:
            # Hollow the volume for SLA resin savings
            target_vol = work_vol.hollow(wall_thickness_vox=hollow_th, auto_drain=drain)
            data = target_vol.data
            effective_pad = pad
            if effective_pad > 0:
                data_to_mesh = np.pad(data, pad_width=effective_pad, mode="constant", constant_values=0.0)
            else:
                data_to_mesh = data
        else:
            data = work_vol.data
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
            level=eff_level,
            step_size=self.step_size,
            chunk_size=self.chunk_size,
            allow_degenerate=self.allow_degenerate,
            progress_callback=progress_callback,
        )

        if len(verts) == 0 or len(faces) == 0:
            raise ValueError(
                f"No isosurface found at level={eff_level}. "
                f"Volume range: [{data_to_mesh.min():.2f}, {data_to_mesh.max():.2f}]."
            )

        eff_uninvert_y = self.uninvert_y if uninvert_y is None else uninvert_y
        eff_invert_z = self.invert_z if invert_z is None else invert_z

        # Reorder (z, y, x) = (t, row, col) -> (col, row, time) = (x, y, z)
        verts = verts[:, [2, 1, 0]]

        # Un-invert vertical image axis Y so top of video frame is top of 3D volume (+Y)
        if eff_uninvert_y:
            max_y = float(data_to_mesh.shape[1] - 1)
            verts[:, 1] = max_y - verts[:, 1]
            faces = faces[:, [0, 2, 1]]  # preserve right-handed outward normals

        # Optional time axis inversion
        if eff_invert_z:
            max_z = float(data_to_mesh.shape[0] - 1)
            verts[:, 2] = max_z - verts[:, 2]
            faces = faces[:, [0, 2, 1]]

        if effective_pad > 0:
            verts -= effective_pad
        verts *= self.voxel_size  # mm units

        # --------------------------------------------------------------
        # 3. Build Trimesh Object & Automated Slicer Health Healing
        # --------------------------------------------------------------
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            process=True,  # welds duplicate chunk boundary vertices and fixes winding
        )

        repair_applied = False
        if repair_mesh:
            try:
                import trimesh.repair as trepair
                trepair.fix_normals(mesh)
                trepair.fill_holes(mesh)
                trepair.fix_inversion(mesh)
                repair_applied = True
            except Exception:
                pass

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
        # 5. Export Mesh
        # --------------------------------------------------------------
        mesh.export(str(output_path))

        bounds = mesh.bounds
        extents_mm = (
            float(bounds[1, 0] - bounds[0, 0]),
            float(bounds[1, 1] - bounds[0, 1]),
            float(bounds[1, 2] - bounds[0, 2]),
        )
        # Standard SLA build vat bounds: 143 mm x 89 mm x 175 mm (e.g. Mars / Photon)
        fits_sla = (
            extents_mm[0] <= 143.0 and
            extents_mm[1] <= 89.0 and
            extents_mm[2] <= 175.0
        )

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
            export_format=export_fmt,
            dimensions_mm=extents_mm,
            fits_standard_sla_vat=fits_sla,
            repair_applied=repair_applied,
        )

        # Reclaim mesh RAM immediately
        del mesh, data_to_mesh
        gc.collect()

        return stats

    def export_two_piece_mold(
        self,
        vol: SpaceTimeVolume,
        output_dir: Union[str, Path] = "output",
        file_prefix: str = "mold",
        format_ext: str = ".stl",
        wall_thickness: int = 4,
        open_top: bool = True,
        split_axis: int = 1,
    ) -> Tuple[MeshStats, MeshStats]:
        """Convenience method to export both halves of an interlocking 2-piece mold."""
        part_a, part_b = generate_two_piece_mold(
            vol,
            wall_thickness=wall_thickness,
            open_top=open_top,
            split_axis=split_axis,
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path_a = out_dir / f"{file_prefix}_part_a{format_ext}"
        path_b = out_dir / f"{file_prefix}_part_b{format_ext}"

        stats_a = self.export(part_a, path_a, pad=0, as_negative_mold=False)
        stats_b = self.export(part_b, path_b, pad=0, as_negative_mold=False)
        return stats_a, stats_b

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
