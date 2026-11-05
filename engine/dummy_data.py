"""
engine/dummy_data.py — Synthetic inputs for out-of-the-box demos.

ALL shapes are built from Signed Distance Fields (SDFs) for smooth gradients.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .sdf_shapes import (
    sdf_disc, sdf_rect, sdf_ring, sdf_capsule,
    sdf_smooth_union, sdf_to_intensity,
)


def dummy_keyframes(h: int = 96, w: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Frame A, Frame B) for Mode 2 (Procedural Generator).

    Both frames are built from SDF primitives -> continuous gradients ->
    marching cubes produces clean, non-staircased isosurfaces.
    """
    edge = 2.5

    # Frame A: rounded rect + disc + capsule link
    sq_a   = sdf_rect(h, w, cy=h*0.28, cx=w*0.25, hh=11, hw=14)
    disc_a = sdf_disc(h, w, cy=h*0.65, cx=w*0.22, r=10)
    cap_a  = sdf_capsule(h, w, ay=h*0.28, ax=w*0.25, by=h*0.65, bx=w*0.22, r=4)
    combined_a = sdf_smooth_union(sdf_smooth_union(sq_a, disc_a, k=6), cap_a, k=4)
    frame_a = sdf_to_intensity(combined_a, edge_width=edge)

    # Frame B: translated to bottom-right
    sq_b   = sdf_rect(h, w, cy=h*0.72, cx=w*0.75, hh=11, hw=14)
    disc_b = sdf_disc(h, w, cy=h*0.35, cx=w*0.78, r=10)
    cap_b  = sdf_capsule(h, w, ay=h*0.72, ax=w*0.75, by=h*0.35, bx=w*0.78, r=4)
    combined_b = sdf_smooth_union(sdf_smooth_union(sq_b, disc_b, k=6), cap_b, k=4)
    frame_b = sdf_to_intensity(combined_b, edge_width=edge)

    return frame_a, frame_b


def dummy_video_array(n: int = 48, h: int = 96, w: int = 128) -> np.ndarray:
    """Synthetic video: orbiting SDF disc + ring over smooth sine field.

    Returns uint8 (n, h, w).
    """
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    out = np.empty((n, h, w), dtype=np.float32)

    for i, phi in enumerate(phase):
        bg = 0.18 + 0.10 * np.sin((xs + ys) / 14.0 + phi * 0.4)

        cy = h / 2.0 + 0.30 * h * np.sin(phi)
        cx = w / 2.0 + 0.32 * w * np.cos(phi)
        d_disc = sdf_disc(h, w, cy, cx, r=10.0)
        disc   = sdf_to_intensity(d_disc, edge_width=1.8)

        cy2 = h / 2.0 + 0.14 * h * np.sin(-phi + np.pi / 3.0)
        cx2 = w / 2.0 + 0.14 * w * np.cos(-phi + np.pi / 3.0)
        d_ring = sdf_ring(h, w, cy2, cx2, r_outer=8.0, r_inner=4.5)
        ring   = sdf_to_intensity(d_ring, edge_width=1.5) * 0.65

        combined = np.clip(bg + 0.70 * disc + ring, 0.0, 1.0)
        out[i] = combined

    return (out * 255.0).astype(np.uint8)


def write_dummy_mp4(
    path: str | Path,
    n: int = 48,
    h: int = 96,
    w: int = 128,
    fps: int = 24,
) -> Path:
    """Write dummy video array to real .mp4 using OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required: pip install opencv-python") from exc

    path = Path(path).resolve()
    frames = dummy_video_array(n, h, w)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h), isColor=False)
    if not vw.isOpened():
        raise IOError(f"OpenCV could not open mp4 writer for {path}.")
    try:
        for frame in frames:
            vw.write(frame)
    finally:
        vw.release()

    return path
