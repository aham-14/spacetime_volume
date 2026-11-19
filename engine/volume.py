"""Core data container. Pure NumPy, no UI / IO dependencies."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


@dataclass
class SpaceTimeVolume:
    """A scalar field sampled on a (t, y, x) grid. Values are float32 in [0, 1]."""

    data: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Expected (t, y, x) array, got shape {self.data.shape}")
        self.data = np.ascontiguousarray(self.data, dtype=np.float32)

    # ---- geometry ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # (t, y, x)

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    def frame(self, t: int) -> np.ndarray:
        """2D slice at integer time index (clamped to valid range)."""
        return self.data[int(np.clip(t, 0, self.n_frames - 1))]

    def sample(self, t: float) -> np.ndarray:
        """2D slice at fractional time via linear interpolation between frames."""
        t = float(np.clip(t, 0, self.n_frames - 1))
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, self.n_frames - 1)
        w = t - i0
        return (1.0 - w) * self.data[i0] + w * self.data[i1]

    # ---- processing -------------------------------------------------------
    def smoothed(
        self,
        sigma_t: float = 1.5,
        sigma_y: float = 0.0,
        sigma_x: float = 0.0,
        truncate: float = 4.0,
    ) -> SpaceTimeVolume:
        """Anisotropic Gaussian blur. A sigma of 0 leaves that axis untouched,
        so (sigma_t, 0, 0) blends frames along time without blurring the image."""
        if max(sigma_t, sigma_y, sigma_x) <= 0:
            return self
        out = gaussian_filter(
            self.data,
            sigma=(sigma_t, sigma_y, sigma_x),
            mode="nearest",
            truncate=truncate,
        )
        return SpaceTimeVolume(
            out, {**self.meta, "smoothing": (sigma_t, sigma_y, sigma_x)}
        )

    def crop(
        self,
        t: Optional[Tuple[int, int]] = None,
        y: Optional[Tuple[int, int]] = None,
        x: Optional[Tuple[int, int]] = None,
        *,
        t0: Optional[int] = None,
        t1: Optional[int] = None,
        y0: Optional[int] = None,
        y1: Optional[int] = None,
        x0: Optional[int] = None,
        x1: Optional[int] = None,
    ) -> SpaceTimeVolume:
        """Extract a 3D sub-volume (ROI) along time and spatial dimensions.

        Ranges can be specified either as tuples `t=(t0, t1)` or keyword args.
        Indices are clamped to array bounds.
        """
        T, H, W = self.shape

        st0 = max(0, t[0] if t is not None else (t0 if t0 is not None else 0))
        st1 = min(T, t[1] if t is not None else (t1 if t1 is not None else T))
        sy0 = max(0, y[0] if y is not None else (y0 if y0 is not None else 0))
        sy1 = min(H, y[1] if y is not None else (y1 if y1 is not None else H))
        sx0 = max(0, x[0] if x is not None else (x0 if x0 is not None else 0))
        sx1 = min(W, x[1] if x is not None else (x1 if x1 is not None else W))

        if st1 <= st0 or sy1 <= sy0 or sx1 <= sx0:
            raise ValueError(
                f"Invalid crop window: t=[{st0}, {st1}), y=[{sy0}, {sy1}), x=[{sx0}, {sx1}) on shape {self.shape}"
            )

        cropped_data = self.data[st0:st1, sy0:sy1, sx0:sx1].copy()
        new_meta = {
            **self.meta,
            "crop_bounds": ((st0, st1), (sy0, sy1), (sx0, sx1)),
            "parent_shape": self.shape,
        }
        return SpaceTimeVolume(cropped_data, new_meta)

    def resize(
        self,
        new_shape: Optional[Tuple[int, int, int]] = None,
        *,
        scale: Optional[Union[float, Tuple[float, float, float]]] = None,
        order: int = 1,
    ) -> SpaceTimeVolume:
        """Resample the volume along time and spatial dimensions using spline interpolation.

        Parameters
        ----------
        new_shape : (new_t, new_y, new_x) tuple or None
        scale : float or (scale_t, scale_y, scale_x) tuple or None
        order : int
            Spline interpolation order (0=nearest, 1=linear/trilinear, 2=quadratic, 3=cubic).
        """
        T, H, W = self.shape
        if new_shape is not None:
            if len(new_shape) != 3 or any(d <= 0 for d in new_shape):
                raise ValueError(f"Invalid new_shape: {new_shape}")
            zoom_factors = (
                float(new_shape[0]) / T,
                float(new_shape[1]) / H,
                float(new_shape[2]) / W,
            )
        elif scale is not None:
            if isinstance(scale, (int, float)):
                zoom_factors = (float(scale), float(scale), float(scale))
            elif len(scale) == 3:
                zoom_factors = (float(scale[0]), float(scale[1]), float(scale[2]))
            else:
                raise ValueError(f"Invalid scale specification: {scale}")
        else:
            raise ValueError("Must provide either new_shape or scale parameter.")

        if all(abs(zf - 1.0) < 1e-6 for zf in zoom_factors):
            return self

        resampled = zoom(self.data, zoom=zoom_factors, order=order, mode="nearest")
        resampled = np.clip(resampled, 0.0, 1.0).astype(np.float32)

        new_meta = {
            **self.meta,
            "resize_factors": zoom_factors,
            "original_shape": self.shape,
        }
        return SpaceTimeVolume(resampled, new_meta)

    def histogram(
        self,
        bins: int = 50,
        value_range: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the scalar field intensity histogram.

        Returns
        -------
        counts : np.ndarray
            1D array of length `bins`
        bin_edges : np.ndarray
            1D array of length `bins + 1`
        """
        counts, bin_edges = np.histogram(self.data, bins=bins, range=value_range)
        return counts, bin_edges

    def motion_mask(
        self,
        threshold: float = 0.05,
        normalize: bool = True,
    ) -> SpaceTimeVolume:
        """Compute temporal difference field isolating dynamic features.

        Calculates |vol[t] - vol[t-1]| and smooths/thresholds it. Static regions
        become 0, keeping only moving pixels for lightweight 'ghost-of-motion'
        3D prints.

        Parameters
        ----------
        threshold : float
            Sensitivity threshold. Changes below this are suppressed.
        normalize : bool
            If True, clips and scales motion gradients into [0, 1].
        """
        T, H, W = self.shape
        if T < 2:
            return self

        diff = np.abs(np.diff(self.data, axis=0))  # (T-1, H, W)
        # Pad first slice with 0 or the first difference to preserve T count
        padded_diff = np.concatenate(
            [np.zeros((1, H, W), dtype=np.float32), diff], axis=0
        )

        if threshold > 0.0:
            if normalize:
                mask = np.clip((padded_diff - threshold) / (1.0 - threshold + 1e-7), 0.0, 1.0)
            else:
                mask = np.where(padded_diff >= threshold, padded_diff, 0.0).astype(np.float32)
        else:
            mask = padded_diff

        new_meta = {
            **self.meta,
            "is_motion_mask": True,
            "motion_threshold": threshold,
        }
        return SpaceTimeVolume(mask.astype(np.float32), new_meta)

    def hollow(
        self,
        wall_thickness_vox: float = 3.0,
        auto_drain: bool = True,
        drain_hole_radius: float = 3.0,
        drain_hole_positions: Optional[List[Tuple[int, int]]] = None,
    ) -> SpaceTimeVolume:
        """Hollow the volume to a uniform shell of `wall_thickness_vox` voxels.

        Uses 3D Euclidean Distance Transform (EDT) on the object interior.
        Voxels located deeper than `wall_thickness_vox` from the outer surface
        are cleared to 0.0 (empty interior cavity).

        If `auto_drain=True`, carves circular vent/drain holes from the bottom
        (z=0) into the cavity so uncured liquid resin can drain during SLA printing.

        Parameters
        ----------
        wall_thickness_vox : float
            Thickness of the solid shell in voxels (e.g. 2.0 to 4.0 voxels).
        auto_drain : bool
            Whether to punch drainage holes through the base.
        drain_hole_radius : float
            Radius of the drainage holes in voxels.
        drain_hole_positions : list of (cy, cx) | None
            Specific (row, col) coordinates for drainage holes on base slices.
            If None and auto_drain=True, automatically computes centroid of base.
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as exc:
            raise ImportError("scipy is required for volume hollowing") from exc

        T, H, W = self.shape
        inside = self.data >= 0.5
        if not np.any(inside):
            return self

        dist = distance_transform_edt(inside)

        # Clear interior cavity
        hollow_arr = self.data.copy()
        cavity_mask = dist > float(wall_thickness_vox)
        hollow_arr[cavity_mask] = 0.0

        # Punch drain holes
        if auto_drain:
            # Find base Z level where object starts
            active_z = np.where(inside.any(axis=(1, 2)))[0]
            if len(active_z) > 0:
                base_z = int(active_z[0])
                # Punch hole from base_z up through the bottom wall
                max_drain_z = min(T, base_z + int(np.ceil(wall_thickness_vox * 2)) + 2)

                if drain_hole_positions is None:
                    # Auto-compute centroid of base slice
                    base_slice = inside[base_z]
                    ys, xs = np.mgrid[0:H, 0:W]
                    denom = float(base_slice.sum())
                    if denom > 0:
                        cy = int((ys * base_slice).sum() / denom)
                        cx = int((xs * base_slice).sum() / denom)
                        holes = [(cy, cx)]
                    else:
                        holes = [(H // 2, W // 2)]
                else:
                    holes = [(int(y), int(x)) for y, x in drain_hole_positions]

                y_grid, x_grid = np.ogrid[:H, :W]
                for cy, cx in holes:
                    hole_mask = (y_grid - cy) ** 2 + (x_grid - cx) ** 2 <= (drain_hole_radius ** 2)
                    for z in range(base_z, max_drain_z):
                        hollow_arr[z, hole_mask] = 0.0

        new_meta = {
            **self.meta,
            "is_hollow": True,
            "wall_thickness_vox": wall_thickness_vox,
            "has_drain_holes": auto_drain,
        }
        return SpaceTimeVolume(hollow_arr, new_meta)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the volume and its metadata to a compressed `.npz` archive."""
        out_path = Path(path)
        if not out_path.suffix:
            out_path = out_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize metadata to JSON string safely
        def _default_serializer(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return str(obj)

        meta_str = json.dumps(self.meta, default=_default_serializer)

        np.savez_compressed(
            out_path,
            data=self.data,
            meta_json=np.array(meta_str),
        )
        return out_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> SpaceTimeVolume:
        """Load a SpaceTimeVolume from a `.npz` file created by `.save()`."""
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Volume file not found: {in_path}")

        with np.load(in_path) as loaded:
            data = loaded["data"]
            meta_raw = loaded["meta_json"]
            meta_str = str(meta_raw.item()) if meta_raw.ndim == 0 else str(meta_raw[0])
            meta = json.loads(meta_str) if meta_str else {}

        return cls(data=data, meta=meta)
