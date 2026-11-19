"""
engine/optical_flow.py — Optical-flow sub-frame interpolation (Mode 1).

WHY OPTICAL FLOW FOR 3D PRINTING:
-----------------------------------
Naive frame stacking inserts a hard intensity discontinuity at every frame
boundary.  When a fast-moving object shifts by N pixels between consecutive
frames, the stacked Z-axis contains a "staircase step" N pixels wide.
Marching cubes amplifies this into a sawtooth mesh.

Farneback optical flow computes a dense 2-D displacement field (u, v) mapping
every pixel in frame A to its estimated position in frame B.  Sub-frame
interpolation synthesises M intermediate frames by:

  1. Warping A *forward*  by fraction t of the flow  → A_warped(t)
  2. Warping B *backward* by fraction (1-t)           → B_warped(t)
  3. Blending: frame(t) = (1-t)·A_warped + t·B_warped

The resulting stack has pixel displacements of ≤1 pixel between consecutive
sub-frames, eliminating Z-axis staircasing.

Design:
  OpticalFlowInterpolator is a pure transformer:
    volume_in  → (raw SpaceTimeVolume, n_frames)
    volume_out → (upsampled SpaceTimeVolume, n_frames * interp_factor)
  It has zero UI dependencies and can be called from FastAPI.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .volume import SpaceTimeVolume


class OpticalFlowInterpolator:
    """Upsample a SpaceTimeVolume along the time axis using Farneback optical flow.

    Parameters
    ----------
    interp_factor : int
        Output frames between (and including) each input frame pair.
        ``interp_factor=1`` → no-op (returns original).
        ``interp_factor=4`` → 3 new sub-frames inserted per pair
                              (time-axis is 4× denser).
    method : str
        Optical flow algorithm:
        - "farneback": Robust classical dense polynomial expansion.
        - "dis": Dense Inverse Search (10–50× faster, highly accurate on gradients).
    pyr_scale : float
        Image pyramid scale between levels (Farneback param, default 0.5).
    levels : int
        Number of pyramid levels (default 3).
    winsize : int
        Averaging window size in pixels (default 15).
        Larger values → smoother flow, less noise, but slower.
    iterations : int
        Iterations per pyramid level (default 3).
    poly_n : int
        Neighbourhood size for polynomial expansion (5 or 7).
    poly_sigma : float
        Gaussian sigma for polynomial expansion (1.1–1.5).
    dis_preset : int
        Preset for DIS flow: 0=ULTRAFAST, 1=FAST, 2=MEDIUM (default 1).
    """

    def __init__(
        self,
        interp_factor: int = 4,
        method: str = "farneback",
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        dis_preset: int = 1,
    ) -> None:
        if interp_factor < 1:
            raise ValueError(f"interp_factor must be >= 1, got {interp_factor}")
        self.interp_factor = interp_factor
        self.method = str(method).lower()
        if self.method not in ("farneback", "dis"):
            raise ValueError(f"Unknown optical flow method '{method}'. Choose 'farneback' or 'dis'.")
        self.dis_preset = int(np.clip(dis_preset, 0, 2))
        self._flow_params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(self, vol: SpaceTimeVolume) -> SpaceTimeVolume:
        """Return a time-upsampled volume.

        Parameters
        ----------
        vol : SpaceTimeVolume
            Input volume, shape (T, H, W).  Values must be float32 [0,1].

        Returns
        -------
        SpaceTimeVolume
            Output shape ((T-1)*interp_factor + 1, H, W) if interp_factor > 1,
            else a copy of *vol*.

        Raises
        ------
        ImportError — if opencv-python is not installed.
        """
        if self.interp_factor == 1:
            return vol

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for optical-flow interpolation. "
                "Install with:  pip install opencv-python"
            ) from exc

        T, H, W = vol.shape
        if T < 2:
            return vol  # nothing to interpolate

        K = self.interp_factor
        # Output has (T-1)*K + 1 frames: original frames + K-1 sub-frames per gap.
        out_T = (T - 1) * K + 1
        out = np.empty((out_T, H, W), dtype=np.float32)

        for i in range(T - 1):
            frame_a = vol.data[i]       # (H, W) float32 [0,1]
            frame_b = vol.data[i + 1]

            # Farneback expects uint8.
            a_u8 = (frame_a * 255.0).astype(np.uint8)
            b_u8 = (frame_b * 255.0).astype(np.uint8)

            # Compute dense displacement field
            if self.method == "dis":
                preset_id = [
                    cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
                    cv2.DISOPTICAL_FLOW_PRESET_FAST,
                    cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                ][self.dis_preset]
                dis_calc = cv2.DISOpticalFlow.create(preset_id)
                flow = dis_calc.calc(a_u8, b_u8, None)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    a_u8, b_u8, None, **self._flow_params
                )  # (H, W, 2): flow[y,x] = (dx, dy) in pixel units

            # Precompute base coordinate grids (shape H×W).
            xs = np.arange(W, dtype=np.float32)[None, :]  # (1, W)
            ys = np.arange(H, dtype=np.float32)[:, None]  # (H, 1)

            flow_dx = flow[:, :, 0]  # (H, W) — horizontal displacement
            flow_dy = flow[:, :, 1]  # (H, W) — vertical  displacement

            for k in range(K):
                dest_idx = i * K + k
                t = k / K   # fractional time in [0, 1)

                if k == 0:
                    out[dest_idx] = frame_a
                    continue

                # Forward warp: move A by t fraction of the flow.
                map_x_fwd = (xs + t * flow_dx).astype(np.float32)
                map_y_fwd = (ys + t * flow_dy).astype(np.float32)
                a_warped = cv2.remap(
                    frame_a, map_x_fwd, map_y_fwd,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                # Backward warp: move B by (1-t) fraction in the opposite direction.
                map_x_bwd = (xs - (1.0 - t) * flow_dx).astype(np.float32)
                map_y_bwd = (ys - (1.0 - t) * flow_dy).astype(np.float32)
                b_warped = cv2.remap(
                    frame_b, map_x_bwd, map_y_bwd,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                # Blend the two warped frames.
                out[dest_idx] = np.add(
                    np.multiply(1.0 - t, a_warped, dtype=np.float32),
                    np.multiply(t,       b_warped, dtype=np.float32),
                )

        # Write final frame.
        out[-1] = vol.data[-1]

        new_meta = {
            **vol.meta,
            "optical_flow": self.method,
            "interp_factor": K,
            "original_n_frames": T,
        }
        return SpaceTimeVolume(np.clip(out, 0.0, 1.0), new_meta)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def flow_magnitude(vol: SpaceTimeVolume, method: str = "farneback") -> np.ndarray:
        """Compute mean flow magnitude between adjacent frames.

        Returns ndarray shape (T-1,) of mean pixel displacements.
        Useful for diagnosing fast-motion regions before interpolation.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for flow_magnitude()")

        T = vol.n_frames
        mags = np.empty(T - 1, dtype=np.float32)
        use_dis = str(method).lower() == "dis"
        dis_calc = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST) if use_dis else None

        for i in range(T - 1):
            a_u8 = (vol.data[i]     * 255).astype(np.uint8)
            b_u8 = (vol.data[i + 1] * 255).astype(np.uint8)
            if use_dis:
                flow = dis_calc.calc(a_u8, b_u8, None)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    a_u8, b_u8, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
            mags[i] = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        return mags

