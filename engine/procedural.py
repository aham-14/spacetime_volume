"""
engine/procedural.py — Mode 2: SDF-based keyframe interpolation → SpaceTimeVolume.

SDF MORPHING FOR 3D PRINTING:
------------------------------
Morphing directly in Signed Distance Field space ensures topological continuity
and smooth sub-pixel gradients without binary step edges.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .easing import EasingFn, get_easing
from .volume import SpaceTimeVolume
from .sdf_shapes import sdf_to_intensity


def intensity_centroid(img: np.ndarray) -> np.ndarray:
    """Return intensity-weighted centroid [cy, cx] of a 2-D image."""
    h, w = img.shape
    total = float(img.sum())
    if total < 1e-12:
        return np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    ys, xs = np.mgrid[0:h, 0:w]
    return np.array([(ys * img).sum() / total, (xs * img).sum() / total])


def _build_sdf_from_intensity(img: np.ndarray) -> np.ndarray:
    """Approximate an SDF from a smooth [0,1] intensity map."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError("scipy is required") from exc

    inside  = img > 0.5
    outside = ~inside
    dist_in  = distance_transform_edt(inside)
    dist_out = distance_transform_edt(outside)
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


class ProceduralGenerator:
    """Build a (t, y, x) SpaceTimeVolume by transitioning Frame A -> Frame B."""

    def __init__(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        displacement: Optional[Tuple[float, float]] = None,
        edge_width: float = 2.5,
        sdf_a: Optional[np.ndarray] = None,
        sdf_b: Optional[np.ndarray] = None,
    ) -> None:
        a = np.asarray(frame_a, dtype=np.float32)
        b = np.asarray(frame_b, dtype=np.float32)

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("frame_a and frame_b must be 2-D (H, W) arrays.")
        if a.shape != b.shape:
            raise ValueError(f"frame_a shape {a.shape} != frame_b shape {b.shape}.")

        self.a = np.clip(a / max(float(a.max()), 1e-9), 0.0, 1.0)
        self.b = np.clip(b / max(float(b.max()), 1e-9), 0.0, 1.0)
        self.edge_width = edge_width

        self.sdf_a = (
            np.asarray(sdf_a, np.float32) if sdf_a is not None
            else _build_sdf_from_intensity(self.a)
        )
        self.sdf_b = (
            np.asarray(sdf_b, np.float32) if sdf_b is not None
            else _build_sdf_from_intensity(self.b)
        )

        if displacement is not None:
            self.displacement = np.asarray(displacement, dtype=float)
        else:
            self.displacement = intensity_centroid(self.b) - intensity_centroid(self.a)

    def generate(
        self,
        n_frames: int = 48,
        transition_function: Union[str, EasingFn] = "linear (t)",
    ) -> SpaceTimeVolume:
        """Generate transition volume via SDF coordinate morphing."""
        if n_frames < 2:
            raise ValueError(f"n_frames must be >= 2, got {n_frames}")

        ease = get_easing(transition_function)
        t_linear = np.linspace(0.0, 1.0, n_frames)
        alpha = np.clip(ease(t_linear), 0.0, 1.0).astype(np.float32)

        dy, dx = float(self.displacement[0]), float(self.displacement[1])
        h, w = self.a.shape
        out = np.empty((n_frames, h, w), dtype=np.float32)

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

        for i, al in enumerate(alpha):
            al = float(al)
            # Sample SDF_A shifted forward by al * displacement
            shifted_a = self._sample_sdf(self.sdf_a, ys - al * dy, xs - al * dx)
            # Sample SDF_B shifted backward by (1-al) * displacement
            shifted_b = self._sample_sdf(self.sdf_b, ys + (1.0 - al) * dy, xs + (1.0 - al) * dx)

            # Linear interpolation in distance field space
            blended_sdf = (1.0 - al) * shifted_a + al * shifted_b
            out[i] = sdf_to_intensity(blended_sdf, edge_width=self.edge_width)

        easing_label = (
            transition_function if isinstance(transition_function, str)
            else getattr(transition_function, "__name__", repr(transition_function))
        )
        return SpaceTimeVolume(
            np.clip(out, 0.0, 1.0),
            {
                "mode": "procedural",
                "easing": easing_label,
                "n_frames": n_frames,
                "displacement_dy": float(dy),
                "displacement_dx": float(dx),
                "edge_width": self.edge_width,
                "morphing": "sdf_interpolation",
            },
        )

    @staticmethod
    def _sample_sdf(sdf: np.ndarray, sample_ys: np.ndarray, sample_xs: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        coords = np.array([
            np.clip(sample_ys, 0, sdf.shape[0] - 1),
            np.clip(sample_xs, 0, sdf.shape[1] - 1),
        ])
        return map_coordinates(sdf, coords, order=1, mode="nearest").astype(np.float32)

    def generate_all(self, n_frames: int = 48) -> dict[str, SpaceTimeVolume]:
        from .easing import EASINGS
        return {name: self.generate(n_frames, name) for name in EASINGS}
