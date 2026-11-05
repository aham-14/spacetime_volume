"""Core data container. Pure NumPy, no UI / IO dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter


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
    def smoothed(self, sigma_t: float = 1.5, sigma_y: float = 0.0,
                 sigma_x: float = 0.0, truncate: float = 4.0) -> "SpaceTimeVolume":
        """Anisotropic Gaussian blur. A sigma of 0 leaves that axis untouched,
        so (sigma_t, 0, 0) blends frames along time without blurring the image."""
        if max(sigma_t, sigma_y, sigma_x) <= 0:
            return self
        out = gaussian_filter(self.data, sigma=(sigma_t, sigma_y, sigma_x),
                              mode="nearest", truncate=truncate)
        return SpaceTimeVolume(out, {**self.meta, "smoothing": (sigma_t, sigma_y, sigma_x)})
