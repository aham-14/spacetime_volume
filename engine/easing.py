"""
engine/easing.py — Transition / easing curves.

Each function maps a NumPy array t ∈ [0, 1] → α ∈ [0, 1].
All are fully vectorised (no Python loops).
"""
from __future__ import annotations

from typing import Callable, Dict, Union

import numpy as np

# Type alias: accepts scalar or ndarray, returns same shape.
EasingFn = Callable[[np.ndarray], np.ndarray]


def _ease_in_out_cubic(t: np.ndarray) -> np.ndarray:
    """Smooth cubic ease-in-out: slow start, fast middle, slow end."""
    return np.where(t < 0.5, 4.0 * t ** 3, 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0)


def _exponential_ease_in(t: np.ndarray) -> np.ndarray:
    """Exponential ease-in; maps [0,1]→[0,1] with f(0)=0, f(1)=1."""
    return (2.0 ** (10.0 * t) - 1.0) / (2.0 ** 10.0 - 1.0)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    """Ken Perlin's smootherstep (6t⁵ − 15t⁴ + 10t³)."""
    return t ** 3 * (t * (6.0 * t - 15.0) + 10.0)


EASINGS: Dict[str, EasingFn] = {
    # --- linear ---
    "linear (t)":               lambda t: np.asarray(t, dtype=float),

    # --- ease-in family ---
    "ease-in quadratic (t²)":   lambda t: np.asarray(t, dtype=float) ** 2,
    "ease-in cubic (t³)":       lambda t: np.asarray(t, dtype=float) ** 3,
    "exponential ease-in":      _exponential_ease_in,

    # --- ease-out family ---
    "ease-out quadratic":       lambda t: 1.0 - (1.0 - np.asarray(t, dtype=float)) ** 2,
    "ease-out cubic":           lambda t: 1.0 - (1.0 - np.asarray(t, dtype=float)) ** 3,

    # --- ease-in-out family ---
    "ease-in-out cubic":        _ease_in_out_cubic,
    "ease-in-out sine":         lambda t: -(np.cos(np.pi * np.asarray(t, dtype=float)) - 1.0) / 2.0,
    "smoothstep (3t²−2t³)":     lambda t: (lambda u: u * u * (3.0 - 2.0 * u))(np.asarray(t, dtype=float)),
    "smootherstep (6t⁵…)":      _smootherstep,
}

# Ordered list of names for UI dropdown
EASING_NAMES = list(EASINGS.keys())


def get_easing(name_or_fn: Union[str, EasingFn]) -> EasingFn:
    """Resolve an easing name or pass through any callable."""
    if callable(name_or_fn):
        return name_or_fn
    if isinstance(name_or_fn, str):
        if name_or_fn not in EASINGS:
            raise KeyError(
                f"Unknown easing: {name_or_fn!r}. Available: {list(EASINGS.keys())}"
            )
        return EASINGS[name_or_fn]
    raise TypeError(f"Expected a string easing name or callable; got {type(name_or_fn)}")
