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
        start_frame: int = 0,
        end_frame: Optional[int] = None,
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
        self.start_frame = max(0, int(start_frame or 0))
        self.end_frame = int(end_frame) if end_frame is not None and end_frame > 0 else None

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

        # Compute start and end frame trimming
        s_frame = min(total_frames, self.start_frame) if total_frames > 0 else 0
        e_frame = min(total_frames, self.end_frame) if self.end_frame is not None and self.end_frame > 0 else total_frames
        if e_frame <= s_frame:
            e_frame = total_frames
        trimmed_count = max(1, e_frame - s_frame)

        # Compute effective stride:
        eff_stride = self.stride
        if self.auto_stride and self.max_frames and self.max_frames > 0 and trimmed_count > self.max_frames:
            eff_stride = max(1, trimmed_count // self.max_frames)

        # Compute effective scale:
        eff_scale = self.scale
        if self.max_dimension and max(orig_w, orig_h) > self.max_dimension:
            eff_scale = min(self.scale, self.max_dimension / float(max(orig_w, orig_h)))

        target_w = int(orig_w * eff_scale) if eff_scale != 1.0 and orig_w > 0 else None
        target_h = int(orig_h * eff_scale) if eff_scale != 1.0 and orig_h > 0 else None

        # Seek to start frame
        if s_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(s_frame))

        frames, idx = [], s_frame
        try:
            while idx < e_frame:
                ok, bgr = cap.read()
                if not ok:
                    break
                if (idx - s_frame) % eff_stride == 0:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    if target_w and target_h:
                        gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    frames.append(gray)
                    if self.max_frames and self.max_frames > 0 and len(frames) >= self.max_frames:
                        break
                idx += 1
        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames decoded from {target_path} in range [{s_frame}, {e_frame})")

        vol = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (t, y, x)
        meta = {
            "source": str(target_path),
            "fps": fps,
            "orig_total_frames": total_frames,
            "orig_resolution": (orig_h, orig_w),
            "start_frame": s_frame,
            "end_frame": e_frame,
            "trimmed_frames": trimmed_count,
            "eff_stride": eff_stride,
            "eff_scale": eff_scale,
            "mode": "video",
        }
        return SpaceTimeVolume(vol, meta)

    @staticmethod
    def probe_video(path: Union[str, Path]) -> dict:
        """Inspect video file metadata without decoding all frames."""
        target_path = Path(path)
        if not target_path.exists():
            return {"exists": False}
        cap = cv2.VideoCapture(str(target_path))
        if not cap.isOpened():
            return {"exists": True, "valid": False}
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {
            "exists": True,
            "valid": True,
            "total_frames": total_frames,
            "width": orig_w,
            "height": orig_h,
            "fps": round(fps, 2),
            "duration_s": round(total_frames / fps, 2) if fps > 0 else 0.0,
        }

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
