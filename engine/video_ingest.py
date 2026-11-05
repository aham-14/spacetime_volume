"""Mode 1: video -> SpaceTimeVolume (OpenCV)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .volume import SpaceTimeVolume


class VideoIngestor:
    """Reads an .mp4, converts to grayscale, stacks to (t, y, x)."""

    def __init__(self, stride: int = 1, max_frames: Optional[int] = None,
                 scale: float = 1.0) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride, self.max_frames, self.scale = stride, max_frames, scale

    def load(self, path: str | Path) -> SpaceTimeVolume:
        path = Path(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames, idx = [], 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    if self.scale != 1.0:
                        gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                                          interpolation=cv2.INTER_AREA)
                    frames.append(gray)
                    if self.max_frames and len(frames) >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"No frames decoded from {path}")
        vol = np.stack(frames, axis=0).astype(np.float32) / 255.0   # (t, y, x)
        return SpaceTimeVolume(vol, {"source": str(path), "fps": fps, "mode": "video"})

    @staticmethod
    def from_array(frames: np.ndarray) -> SpaceTimeVolume:
        """Wrap an already-decoded (t, y, x) uint8/float array (e.g. dummy data)."""
        arr = np.asarray(frames)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        return SpaceTimeVolume(arr, {"source": "array", "mode": "video"})
