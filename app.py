"""
app.py — Composition root for the Space-Time Volume 3D-print & casting pipeline.

This is the ONLY file that knows about BOTH the engine and the viewer.
In a web-app context, this layer is replaced by FastAPI/Flask endpoint handlers.

Pipeline responsibilities:
  • Mode 1: VideoIngestor → OpticalFlowInterpolator → Gaussian polish
  • Mode 2: ProceduralGenerator (SDF morphing via easing curves)
  • Export Module: MeshExporter with:
      - Negative Mold boolean subtraction (hollowed STL for pouring wax/silicone)
      - High-resolution SLA tuning (step_size=1, reduced Taubin passes)
      - Chunked marching cubes for strict 8GB RAM / 4GB GPU memory budgeting
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from engine import (
    EASING_NAMES,
    OpticalFlowInterpolator,
    ProceduralGenerator,
    VideoIngestor,
    MeshExporter,
    MeshStats,
    dummy_keyframes,
    dummy_video_array,
    write_dummy_mp4,
)
from viewer import SpaceTimeViewer


def build_viewer(
    video_path: Optional[str] = None,
    sigma_t: float = 0.8,
    use_optical_flow: bool = True,
    interp_factor: int = 4,
    n_proc_frames: int = 48,
    edge_width: float = 2.5,
    colorscale: str = "Gray",
    height_px: int = 640,
    stl_output_dir: str = "output",
    stl_step_size: int = 1,
    stl_smooth_iters: int = 4,
    voxel_size_mm: float = 0.5,
    chunk_size: int = 32,
) -> SpaceTimeViewer:
    """Wire backend engine to Plotly/ipywidgets viewer.

    Parameters
    ----------
    video_path : str | None
        Path to an .mp4 file. If None, uses synthetic dummy data.
    sigma_t : float
        Final Gaussian polish sigma along time axis (Mode 1).
    use_optical_flow : bool
        Whether to apply Farneback optical flow frame interpolation (Mode 1).
    interp_factor : int
        Sub-frame density for optical flow (4 = 4x time resolution).
    n_proc_frames : int
        Number of frames in Mode 2 procedural transition.
    edge_width : float
        SDF edge softness in pixels for smooth isosurface extraction.
    colorscale : str
        Plotly surface colorscale.
    height_px : int
        Viewer height in pixels.
    stl_output_dir : str
        Default directory for STL exports.
    stl_step_size : int
        Marching cubes step size (1 = max SLA resin resolution).
    stl_smooth_iters : int
        Taubin smoothing iterations (default 4 for SLA resin).
    voxel_size_mm : float
        Physical voxel size in mm.
    chunk_size : int
        Z-slice chunk depth for marching cubes (keeps RAM strictly <= 8GB).

    Returns
    -------
    SpaceTimeViewer
    """

    # ------------------------------------------------------------------
    # State cells for interactive UI reactivity
    # ------------------------------------------------------------------
    _sigma_cell     = [sigma_t]
    _use_flow_cell  = [use_optical_flow]
    _raw_vol_cache  = [None]

    # ------------------------------------------------------------------
    # Mode 1 Factory — Video + Optical Flow + Polish
    # ------------------------------------------------------------------
    def _load_raw_video():
        if _raw_vol_cache[0] is not None:
            return _raw_vol_cache[0]
        ingestor = VideoIngestor(stride=1, max_frames=120, scale=1.0)
        if video_path is not None:
            raw = ingestor.load(video_path)
        else:
            try:
                tmp_path = Path(tempfile.gettempdir()) / "spacetime_dummy.mp4"
                write_dummy_mp4(tmp_path)
                raw = ingestor.load(tmp_path)
            except (ImportError, IOError):
                raw = VideoIngestor.from_array(dummy_video_array(n=48))
        _raw_vol_cache[0] = raw
        return raw

    def video_factory():
        raw = _load_raw_video()
        vol = raw
        if _use_flow_cell[0]:
            try:
                interpolator = OpticalFlowInterpolator(interp_factor=interp_factor)
                vol = interpolator.interpolate(vol)
            except ImportError:
                pass  # Graceful fallback if cv2 not available

        return vol.smoothed(sigma_t=_sigma_cell[0], sigma_y=0.0, sigma_x=0.0)

    # ------------------------------------------------------------------
    # Mode 2 Factory — SDF Procedural Morphing
    # ------------------------------------------------------------------
    frame_a, frame_b = dummy_keyframes()
    _sdf_gen = [None]

    def _get_gen(ew: float) -> ProceduralGenerator:
        if _sdf_gen[0] is None or _sdf_gen[0].edge_width != ew:
            _sdf_gen[0] = ProceduralGenerator(frame_a, frame_b, edge_width=ew)
        return _sdf_gen[0]

    def procedural_factory(easing_name: str, ew: float):
        return _get_gen(ew).generate(n_frames=n_proc_frames, transition_function=easing_name)

    # ------------------------------------------------------------------
    # Mesh Export Factory — SLA & Negative Mold Boolean Subtraction
    # ------------------------------------------------------------------
    def mesh_export_factory(vol, params: dict) -> MeshStats:
        path = params.get("path", "output/model.stl")
        is_mold = params.get("as_negative_mold", False)
        open_top = params.get("mold_open_top", True)
        wall_th = params.get("mold_wall_thickness", 4)
        step_sz = params.get("step_size", stl_step_size)
        smooth_it = params.get("smooth_iterations", stl_smooth_iters)
        vox_sz = params.get("voxel_size", voxel_size_mm)
        chk_sz = params.get("chunk_size", chunk_size)
        iso_level = float(params.get("level", 0.5))
        prog_cb = params.get("progress_callback", None)

        exporter = MeshExporter(
            level=iso_level,
            step_size=step_sz,
            smooth_iterations=smooth_it,
            voxel_size=vox_sz,
            chunk_size=chk_sz,
            as_negative_mold=is_mold,
            mold_wall_thickness=wall_th,
            mold_open_top=open_top,
        )
        return exporter.export(vol, path, level=iso_level, progress_callback=prog_cb)

    # ------------------------------------------------------------------
    # Assemble Viewer & Bind Reactive Controls
    # ------------------------------------------------------------------
    viewer = SpaceTimeViewer(
        video_factory=video_factory,
        procedural_factory=procedural_factory,
        mesh_export_factory=mesh_export_factory,
        easing_names=EASING_NAMES,
        initial_easing="linear (t)",
        colorscale=colorscale,
        height_px=height_px,
        stl_output_dir=stl_output_dir,
    )

    def _on_sigma_change(change):
        _sigma_cell[0] = float(change["new"])
        viewer._volumes["video"] = None
        viewer._activate("video")

    def _on_flow_change(change):
        _use_flow_cell[0] = bool(change["new"])
        viewer._volumes["video"] = None
        viewer._activate("video")

    viewer.sigma_slider.observe(_on_sigma_change, names="value")
    viewer.flow_toggle.observe(_on_flow_change, names="value")

    return viewer
