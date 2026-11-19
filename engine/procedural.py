"""
engine/procedural.py — Mode 2: SDF-based keyframe interpolation → SpaceTimeVolume.

SDF MORPHING FOR 3D PRINTING:
------------------------------
Morphing directly in Signed Distance Field space ensures topological continuity
and smooth sub-pixel gradients without binary step edges.

Multi-Keyframe Support:
-----------------------
Supports arbitrary sequences of 2 or more keyframes. Transitions smoothly through
all keyframes across the generated volume time axis using piecewise coordinate
morphing and easing curves.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

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

    inside = img > 0.5
    outside = ~inside
    dist_in = distance_transform_edt(inside)
    dist_out = distance_transform_edt(outside)
    sdf = dist_out - dist_in
    return sdf.astype(np.float32)


class ProceduralGenerator:
    """Build a (t, y, x) SpaceTimeVolume by transitioning across keyframes in SDF space."""

    def __init__(
        self,
        frame_a: Optional[np.ndarray] = None,
        frame_b: Optional[np.ndarray] = None,
        keyframes: Optional[Sequence[np.ndarray]] = None,
        displacement: Optional[Tuple[float, float]] = None,
        edge_width: float = 2.5,
        sdf_a: Optional[np.ndarray] = None,
        sdf_b: Optional[np.ndarray] = None,
        sdfs: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        # Collect keyframes
        if keyframes is not None:
            raw_kfs = [np.asarray(kf, dtype=np.float32) for kf in keyframes]
        elif frame_a is not None and frame_b is not None:
            raw_kfs = [np.asarray(frame_a, dtype=np.float32), np.asarray(frame_b, dtype=np.float32)]
        else:
            raise ValueError("Must provide either (frame_a, frame_b) or keyframes sequence.")

        if len(raw_kfs) < 2:
            raise ValueError(f"At least 2 keyframes are required, got {len(raw_kfs)}.")

        ref_shape = raw_kfs[0].shape
        if len(ref_shape) != 2:
            raise ValueError(f"Keyframes must be 2-D (H, W), got shape {ref_shape}.")
        for idx, kf in enumerate(raw_kfs):
            if kf.shape != ref_shape:
                raise ValueError(
                    f"Keyframe {idx} shape {kf.shape} does not match keyframe 0 shape {ref_shape}."
                )

        self.keyframes = [
            np.clip(kf / max(float(kf.max()), 1e-9), 0.0, 1.0) for kf in raw_kfs
        ]
        self.a = self.keyframes[0]
        self.b = self.keyframes[-1]
        self.edge_width = float(edge_width)

        # Build or store SDF representations
        if sdfs is not None:
            self.sdfs = [np.asarray(s, dtype=np.float32) for s in sdfs]
        else:
            computed_sdfs = []
            for i, kf in enumerate(self.keyframes):
                if i == 0 and sdf_a is not None:
                    computed_sdfs.append(np.asarray(sdf_a, dtype=np.float32))
                elif i == len(self.keyframes) - 1 and sdf_b is not None:
                    computed_sdfs.append(np.asarray(sdf_b, dtype=np.float32))
                else:
                    computed_sdfs.append(_build_sdf_from_intensity(kf))
            self.sdfs = computed_sdfs

        self.sdf_a = self.sdfs[0]
        self.sdf_b = self.sdfs[-1]

        # Inter-keyframe spatial displacement vectors
        n_segments = len(self.keyframes) - 1
        if displacement is not None and n_segments == 1:
            disp_arr = np.asarray(displacement, dtype=float)
            self.displacements = [disp_arr]
        else:
            self.displacements = [
                intensity_centroid(self.keyframes[i + 1]) - intensity_centroid(self.keyframes[i])
                for i in range(n_segments)
            ]
        self.displacement = self.displacements[0]

    def generate(
        self,
        n_frames: int = 48,
        transition_function: Union[str, EasingFn] = "linear (t)",
    ) -> SpaceTimeVolume:
        """Generate transition volume via piecewise SDF coordinate morphing.

        Parameters
        ----------
        n_frames : int
            Total frame depth along the time (Z) axis (must be >= len(keyframes)).
        transition_function : str | Callable
            Mathematical easing curve name or callable mapping [0, 1] -> [0, 1].
        """
        if n_frames < len(self.keyframes):
            raise ValueError(
                f"n_frames ({n_frames}) must be >= number of keyframes ({len(self.keyframes)})"
            )

        ease = get_easing(transition_function)
        h, w = self.a.shape
        out = np.empty((n_frames, h, w), dtype=np.float32)

        n_segments = len(self.keyframes) - 1
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

        # Global time parameter in [0, 1]
        t_global = np.linspace(0.0, 1.0, n_frames)

        for i, tg in enumerate(t_global):
            # Map global time tg in [0, 1] to continuous segment coordinate in [0, n_segments]
            s = tg * n_segments
            seg_idx = min(int(np.floor(s)), n_segments - 1)
            # Local normalized time inside segment in [0, 1]
            if seg_idx == n_segments - 1 and tg >= 1.0 - 1e-9:
                u = 1.0
            else:
                u = float(np.clip(s - seg_idx, 0.0, 1.0))

            # Apply mathematical easing curve to local progression
            al = float(np.clip(ease(u), 0.0, 1.0))

            sdf_from = self.sdfs[seg_idx]
            sdf_to = self.sdfs[seg_idx + 1]
            dy = float(self.displacements[seg_idx][0])
            dx = float(self.displacements[seg_idx][1])

            # Forward-shift start SDF and backward-shift end SDF along spatial trajectory
            shifted_from = self._sample_sdf(sdf_from, ys - al * dy, xs - al * dx)
            shifted_to = self._sample_sdf(sdf_to, ys + (1.0 - al) * dy, xs + (1.0 - al) * dx)

            # Continuous blend in signed distance space
            blended_sdf = (1.0 - al) * shifted_from + al * shifted_to
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
                "n_keyframes": len(self.keyframes),
                "edge_width": self.edge_width,
                "morphing": "multi_keyframe_sdf_interpolation",
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
