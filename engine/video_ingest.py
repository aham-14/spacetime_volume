"""Mode 1: video -> SpaceTimeVolume (OpenCV)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .volume import SpaceTimeVolume


class VideoIngestor:
    """Reads video files (.mp4, .mov, .avi, etc.), converts to luminance, and stacks into (t, y, x).

    Features smart frame striding and max dimension scaling to prevent OOM
    and compute delays on high-resolution or long real-world video files.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path, int]] = None,
        stride: int = 1,
        max_frames: Optional[int] = None,
        scale: float = 1.0,
        max_dimension: Optional[int] = None,
        auto_stride: bool = True,
    ) -> None:
        if isinstance(path, (int, float)):
            self.default_path = None
            self.stride = int(path)
        elif isinstance(path, (str, Path)):
            self.default_path = Path(path)
            self.stride = stride
        else:
            self.default_path = None
            self.stride = stride

        if self.stride < 1:
            self.stride = 1

        self.max_frames = max_frames
        self.scale = float(scale)
        self.max_dimension = max_dimension
        self.auto_stride = auto_stride

    def load(self, path: Optional[Union[str, Path]] = None) -> SpaceTimeVolume:
        """Decode video file into a continuous (t, y, x) SpaceTimeVolume."""
        target_path = Path(path if path is not None else self.default_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Video file not found: {target_path}")

        cap = cv2.VideoCapture(str(target_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {target_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Compute effective stride:
        # If auto_stride is True and max_frames is set, sample uniformly across entire video
        eff_stride = self.stride
        if self.auto_stride and self.max_frames and total_frames > self.max_frames:
            eff_stride = max(1, total_frames // self.max_frames)

        # Compute effective scale:
        eff_scale = self.scale
        if self.max_dimension and max(orig_w, orig_h) > self.max_dimension:
            eff_scale = min(self.scale, self.max_dimension / float(max(orig_w, orig_h)))

        target_w = int(orig_w * eff_scale) if eff_scale != 1.0 and orig_w > 0 else None
        target_h = int(orig_h * eff_scale) if eff_scale != 1.0 and orig_h > 0 else None

        frames, idx = [], 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % eff_stride == 0:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    if target_w and target_h:
                        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    frames.append(gray)
                    if self.max_frames and len(frames) >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames decoded from {target_path}")

        vol = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (t, y, x)
        meta = {
            "source": str(target_path),
            "fps": fps,
            "orig_total_frames": total_frames,
            "orig_resolution": (orig_h, orig_w),
            "eff_stride": eff_stride,
            "eff_scale": eff_scale,
            "mode": "video",
        }
        return SpaceTimeVolume(vol, meta)

    def ingest(self, path: Optional[Union[str, Path]] = None) -> SpaceTimeVolume:
        """Alias for load()."""
        return self.load(path)

    @staticmethod
    def from_array(frames: np.ndarray) -> SpaceTimeVolume:
        """Wrap an already-decoded (t, y, x) uint8/float array (e.g. dummy data)."""
        arr = np.asarray(frames)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        return SpaceTimeVolume(arr, {"source": "array", "mode": "video"})
