"""
viewer.py — 3-D Interactive Space-Time viewer for SLA Resin & Casting Workflows.

Technology: plotly.graph_objects.FigureWidget + ipywidgets.

FEATURES:
  • Single flat `go.Surface` plane rendering projected frames using `surfacecolor`.
  • Dynamic time slider scrubbing along the Z-axis.
  • Mode 1 & 2 switching with Farneback optical flow & SDF edge width tuning.
  • Negative Mold generation toggle (Boolean subtraction for pouring wax/silicone).
  • High-Resolution SLA step_size control & low Taubin smoothing presets.
  • Chunked processing & memory management for 8GB RAM / 4GB GPU systems.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional, Sequence

import ipywidgets as w
import numpy as np
import plotly.graph_objects as go

from engine import SpaceTimeVolume, MeshStats

# ------------------------------------------------------------------
# Factory type aliases
# ------------------------------------------------------------------
VolumeFactory     = Callable[[], SpaceTimeVolume]
ProceduralFactory = Callable[[str, float], SpaceTimeVolume]    # (easing, edge_width) -> vol
MeshExportFactory = Callable[[SpaceTimeVolume, dict], MeshStats]  # (vol, params) -> MeshStats


class SpaceTimeViewer:
    """Interactive viewer + SLA/Mold STL export panel for Space-Time Volumes."""

    def __init__(
        self,
        video_factory: VolumeFactory,
        procedural_factory: ProceduralFactory,
        mesh_export_factory: MeshExportFactory,
        easing_names: Sequence[str],
        initial_easing: Optional[str] = None,
        colorscale: str = "Gray",
        height_px: int = 640,
        stl_output_dir: str = "output",
    ) -> None:
        self._video_factory   = video_factory
        self._proc_factory    = procedural_factory
        self._export_factory  = mesh_export_factory
        self._colorscale      = colorscale
        self._stl_output_dir  = stl_output_dir

        # Volume cache
        self._volumes: dict[str, Optional[SpaceTimeVolume]] = {
            "video": None, "procedural": None
        }
        self.volume: Optional[SpaceTimeVolume] = None
        self._xy_cache: Optional[tuple] = None

        self._build_controls(list(easing_names), initial_easing)
        self._build_figure(height_px)
        self._build_layout()
        self._wire_callbacks()
        self._activate("video")

    # ==================================================================
    # Widget construction
    # ==================================================================

    def _build_controls(self, easing_names: list[str], initial_easing: Optional[str]) -> None:
        # --- Mode selector ---
        self.mode_toggle = w.ToggleButtons(
            options=[("🎥  Mode 1 — Video (Optical Flow)", "video"),
                     ("⚡  Mode 2 — SDF Procedural",        "procedural")],
            value="video",
            description="Mode:",
            layout=w.Layout(margin="4px 0"),
        )

        # --- Easing dropdown (Mode 2) ---
        self.easing_dropdown = w.Dropdown(
            options=easing_names,
            value=initial_easing or easing_names[0],
            description="Easing:",
            disabled=True,
            layout=w.Layout(width="360px"),
        )

        # --- SDF edge width (Mode 2) ---
        self.edge_width_slider = w.FloatSlider(
            value=2.5, min=0.5, max=8.0, step=0.5,
            description="SDF edge σ:",
            tooltip="Sub-pixel gradient softness for smooth isosurface (px)",
            continuous_update=False,
            readout_format=".1f",
            disabled=True,
            layout=w.Layout(width="300px"),
        )

        # --- Optical flow toggle (Mode 1) ---
        self.flow_toggle = w.Checkbox(
            value=True,
            description="Farneback optical flow (4× sub-frames)",
            tooltip="Interpolate sub-frames with dense optical flow to eliminate Z-staircasing",
            indent=False,
            layout=w.Layout(width="300px"),
        )

        # --- Temporal sigma (Mode 1 polish) ---
        self.sigma_slider = w.FloatSlider(
            value=0.8, min=0.0, max=4.0, step=0.25,
            description="σ_t polish:",
            tooltip="Light Gaussian polish along time axis",
            continuous_update=False,
            readout_format=".2f",
            layout=w.Layout(width="280px"),
        )

        # --- Time scrubber ---
        self.time_slider = w.IntSlider(
            value=0, min=0, max=1, step=1,
            description="Time (Z):",
            continuous_update=True,
            layout=w.Layout(width="590px"),
        )
        self.play_btn = w.Play(value=0, min=0, max=1, step=1, interval=55)
        w.jslink((self.play_btn, "value"), (self.time_slider, "value"))

        # --- SLA & Negative Mold Export Controls ---
        self.as_mold_toggle = w.Checkbox(
            value=False,
            description="📦 Export Negative Mold (Boolean Subtraction for Wax/Silicone Casting)",
            tooltip="Subtracts the volume from a solid bounding box to create a pouring mold",
            indent=False,
            layout=w.Layout(width="520px"),
        )
        self.mold_open_top = w.Checkbox(
            value=True,
            description="Open Pour Spout",
            tooltip="Leaves top cavity open for pouring liquid wax or silicone",
            indent=False,
            disabled=True,
            layout=w.Layout(width="160px"),
        )
        self.mold_wall_slider = w.IntSlider(
            value=4, min=2, max=16, step=1,
            description="Wall (vox):",
            tooltip="Mold bounding wall thickness in voxels",
            disabled=True,
            layout=w.Layout(width="220px"),
        )

        self.stl_path_text = w.Text(
            value=str(Path(self._stl_output_dir) / "spacetime_model.stl"),
            description="STL path:",
            layout=w.Layout(width="380px"),
        )
        self.step_size_slider = w.IntSlider(
            value=1, min=1, max=4, step=1,
            description="Step Size (SLA):",
            tooltip="Marching cubes step size: 1 = Max resolution for high-res SLA resin",
            layout=w.Layout(width="260px"),
        )
        self.smooth_iters_slider = w.IntSlider(
            value=4, min=0, max=25, step=1,
            description="Taubin iters:",
            tooltip="Taubin passes (default 4 for SLA resin to keep pixel details sharp)",
            layout=w.Layout(width="240px"),
        )
        self.voxel_size_slider = w.FloatSlider(
            value=0.5, min=0.05, max=2.0, step=0.05,
            description="Voxel (mm):",
            tooltip="Physical size of voxel for real-world print scale",
            readout_format=".2f",
            layout=w.Layout(width="240px"),
        )
        self.chunk_size_slider = w.IntSlider(
            value=32, min=8, max=128, step=8,
            description="Chunk slices:",
            tooltip="Chunk depth for marching cubes (keeps RAM strictly within 8GB)",
            layout=w.Layout(width="240px"),
        )

        self.stl_btn = w.Button(
            description="🖨  Generate STL",
            button_style="success",
            icon="cube",
            layout=w.Layout(width="170px", height="36px"),
        )
        self.stl_progress = w.IntProgress(
            value=0, min=0, max=100,
            bar_style="info",
            layout=w.Layout(width="200px", visibility="hidden"),
        )

        self.regen_btn = w.Button(
            description="↺  Regenerate",
            button_style="info",
            icon="refresh",
            disabled=True,
            layout=w.Layout(width="150px"),
        )

        self.status_html = w.HTML(
            value="<i style='color:#aaa'>Initialising…</i>",
            layout=w.Layout(margin="2px 0"),
        )
        self.stl_status_html = w.HTML(value="")

    def _build_figure(self, height_px: int) -> None:
        surface = go.Surface(
            x=[0, 1], y=[0, 1],
            z=[[0, 0], [0, 0]],
            surfacecolor=[[0.0, 0.0], [0.0, 0.0]],
            colorscale=self._colorscale,
            cmin=0.0, cmax=1.0,
            showscale=True,
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0),
            colorbar=dict(
                title=dict(text="intensity", side="right"),
                thickness=14, len=0.65, x=1.01,
            ),
            hovertemplate="x: %{x}<br>y: %{y}<br>t: %{z:.0f}<br>intensity: %{surfacecolor:.3f}<extra></extra>",
        )
        self.fig = go.FigureWidget(data=[surface])
        self.fig.update_layout(
            height=height_px,
            margin=dict(l=0, r=10, t=40, b=0),
            title=dict(text="Space-Time Volume", x=0.5, font=dict(size=15)),
            scene=dict(
                xaxis=dict(title="x (cols)",  showgrid=True, gridcolor="#444"),
                yaxis=dict(title="y (rows)",  showgrid=True, gridcolor="#444", autorange="reversed"),
                zaxis=dict(title="t (frame)", showgrid=True, gridcolor="#444"),
                aspectmode="manual",
                aspectratio=dict(x=1.4, y=1.0, z=0.8),
                bgcolor="rgb(14,14,20)",
            ),
            paper_bgcolor="rgb(20,20,28)",
            font=dict(color="#ddd"),
        )

    def _build_layout(self) -> None:
        r_mode = w.HBox([self.mode_toggle], layout=w.Layout(margin="4px 0"))

        self._row_mode1 = w.HBox(
            [self.flow_toggle, self.sigma_slider],
            layout=w.Layout(align_items="center", gap="16px", margin="2px 0"),
        )
        self._row_mode2 = w.HBox(
            [self.easing_dropdown, self.edge_width_slider, self.regen_btn],
            layout=w.Layout(align_items="center", gap="16px", margin="2px 0", visibility="hidden"),
        )

        r_time = w.HBox(
            [self.play_btn, self.time_slider],
            layout=w.Layout(align_items="center", gap="8px"),
        )

        # Hardware STL / Negative Mold Box
        r_mold_cfg = w.HBox(
            [self.as_mold_toggle, self.mold_open_top, self.mold_wall_slider],
            layout=w.Layout(align_items="center", gap="12px"),
        )
        r_stl_params = w.HBox(
            [self.stl_path_text, self.step_size_slider, self.smooth_iters_slider,
             self.voxel_size_slider, self.chunk_size_slider],
            layout=w.Layout(align_items="center", gap="14px", flex_wrap="wrap"),
        )
        r_stl_btn = w.HBox(
            [self.stl_btn, self.stl_progress, self.stl_status_html],
            layout=w.Layout(align_items="center", gap="12px"),
        )
        stl_panel = w.VBox(
            [r_mold_cfg, r_stl_params, r_stl_btn],
            layout=w.Layout(
                border="1px solid #3a75c4", border_radius="6px",
                padding="10px 14px", margin="8px 0", background_color="#181e29"
            ),
        )

        sep = w.HTML("<hr style='border-color:#444;margin:4px 0'>")

        self.ui = w.VBox(
            [r_mode, self._row_mode1, self._row_mode2, r_time,
             self.status_html, sep, stl_panel, self.fig],
            layout=w.Layout(padding="8px"),
        )

    def _wire_callbacks(self) -> None:
        self.mode_toggle.observe(self._on_mode_change, names="value")
        self.easing_dropdown.observe(self._on_easing_change, names="value")
        self.time_slider.observe(self._on_slider_change, names="value")
        # NOTE: sigma_slider and flow_toggle are intentionally NOT wired here.
        # app.py's build_viewer() attaches the authoritative observers for those
        # two widgets so it can update _sigma_cell / _use_flow_cell first, then
        # call _activate().  Wiring them here too would cause a double build
        # (once with the stale cell value, once with the updated value).
        self.edge_width_slider.observe(self._on_edge_width_change, names="value")
        self.as_mold_toggle.observe(self._on_mold_toggle, names="value")
        self.regen_btn.on_click(self._on_regen_click)
        self.stl_btn.on_click(self._on_stl_click)

    # ==================================================================
    # Volume lifecycle
    # ==================================================================

    def _activate(self, mode: str) -> None:
        is_proc = mode == "procedural"
        self._row_mode1.layout.visibility = "hidden" if is_proc else "visible"
        self._row_mode2.layout.visibility = "visible" if is_proc else "hidden"

        self.easing_dropdown.disabled    = not is_proc
        self.edge_width_slider.disabled  = not is_proc
        self.regen_btn.disabled          = not is_proc
        self.sigma_slider.disabled       = is_proc
        self.flow_toggle.disabled        = is_proc

        if self._volumes[mode] is None:
            self._status("⏳  Building volume…")
            if mode == "video":
                self._volumes[mode] = self._video_factory()
            else:
                self._volumes[mode] = self._proc_factory(
                    self.easing_dropdown.value,
                    float(self.edge_width_slider.value),
                )
        self._mount_volume(self._volumes[mode])

    def _mount_volume(self, vol: SpaceTimeVolume) -> None:
        self.volume = vol
        t, h, wd = vol.shape

        if self._xy_cache is None or self._xy_cache[0] != (h, wd):
            self._xy_cache = (
                (h, wd),
                np.arange(wd, dtype=np.float32),
                np.arange(h,  dtype=np.float32),
            )

        kept_t = int(np.clip(self.time_slider.value, 0, t - 1))
        self.time_slider.max = t - 1
        self.play_btn.max    = t - 1
        self.time_slider.value = kept_t
        self.play_btn.value    = kept_t

        with self.fig.batch_update():
            self.fig.layout.scene.xaxis.range = [0, wd - 1]
            self.fig.layout.scene.yaxis.range = [h - 1, 0]
            self.fig.layout.scene.zaxis.range = [0, t - 1]

        self._render(kept_t, full=True)
        self._update_status()

    # ==================================================================
    # Rendering
    # ==================================================================

    def _render(self, t_idx: int, full: bool = False) -> None:
        _, X, Y = self._xy_cache
        h, wd = self.volume.shape[1], self.volume.shape[2]

        with self.fig.batch_update():
            surf = self.fig.data[0]
            if full:
                surf.x = X
                surf.y = Y
            surf.z = np.full((h, wd), float(t_idx), dtype=np.float32)
            surf.surfacecolor = self.volume.frame(t_idx)
            self.fig.layout.title.text = (
                f"Space-Time Volume — t = {t_idx} / {self.volume.n_frames - 1}"
            )

    # ==================================================================
    # Callbacks
    # ==================================================================

    def _on_slider_change(self, change: dict) -> None:
        if self.volume is not None:
            self._render(int(change["new"]))

    def _on_mode_change(self, change: dict) -> None:
        self._activate(change["new"])

    def _on_easing_change(self, change: dict) -> None:
        if self.mode_toggle.value != "procedural":
            return
        self._status(f"⏳  Regenerating: easing={change['new']!r}…")
        self._volumes["procedural"] = self._proc_factory(
            change["new"], float(self.edge_width_slider.value)
        )
        self._mount_volume(self._volumes["procedural"])

    def _on_edge_width_change(self, change: dict) -> None:
        if self.mode_toggle.value != "procedural":
            return
        self._status(f"⏳  Regenerating: SDF edge σ={change['new']:.1f}…")
        self._volumes["procedural"] = self._proc_factory(
            self.easing_dropdown.value, float(change["new"])
        )
        self._mount_volume(self._volumes["procedural"])

    def _on_sigma_change(self, change: dict) -> None:
        if self.mode_toggle.value != "video":
            return
        self._volumes["video"] = None
        self._activate("video")

    def _on_flow_toggle(self, change: dict) -> None:
        if self.mode_toggle.value != "video":
            return
        self._volumes["video"] = None
        self._activate("video")

    def _on_mold_toggle(self, change: dict) -> None:
        is_mold = bool(change["new"])
        self.mold_open_top.disabled = not is_mold
        self.mold_wall_slider.disabled = not is_mold
        # Update suggested filename
        base_name = "spacetime_mold.stl" if is_mold else "spacetime_model.stl"
        self.stl_path_text.value = str(Path(self._stl_output_dir) / base_name)

    def _on_regen_click(self, _btn) -> None:
        self._volumes["procedural"] = None
        self._activate("procedural")

    def _on_stl_click(self, _btn) -> None:
        """Trigger background mesh export with SLA / Negative Mold tuning."""
        if self.volume is None:
            self.stl_status_html.value = "<span style='color:#f88'>⚠ No volume loaded.</span>"
            return

        path = self.stl_path_text.value.strip() or "output/model.stl"
        export_params = {
            "path": path,
            "as_negative_mold": bool(self.as_mold_toggle.value),
            "mold_open_top": bool(self.mold_open_top.value),
            "mold_wall_thickness": int(self.mold_wall_slider.value),
            "step_size": int(self.step_size_slider.value),
            "smooth_iterations": int(self.smooth_iters_slider.value),
            "voxel_size": float(self.voxel_size_slider.value),
            "chunk_size": int(self.chunk_size_slider.value),
        }

        self.stl_btn.disabled = True
        self.stl_progress.layout.visibility = "visible"
        self.stl_progress.value = 10
        target_desc = "Negative Mold (Boolean Subtraction)" if export_params["as_negative_mold"] else "SLA Mesh"
        self.stl_status_html.value = f"<span style='color:#fa0'>⏳  Processing {target_desc} (Chunked SLA Extraction)…</span>"

        vol_snapshot = self.volume

        def _run():
            try:
                def _prog(pct):
                    self.stl_progress.value = int(10 + pct * 0.8)

                export_params["progress_callback"] = _prog
                stats = self._export_factory(vol_snapshot, export_params)
                self.stl_progress.value = 100

                wt = "✅ watertight" if stats.is_watertight else "⚠ not watertight"
                mold_lbl = " [Negative Mold]" if stats.is_negative_mold else " [Positive Model]"
                self.stl_status_html.value = (
                    f"<b style='color:#4f4'>✔ STL exported{mold_lbl}</b> → "
                    f"<code>{stats.output_path}</code><br>"
                    f"Verts: {stats.n_vertices:,} &nbsp;|&nbsp; "
                    f"Faces: {stats.n_faces:,} &nbsp;|&nbsp; "
                    f"Vol: {stats.volume_mm3:.1f} mm³ &nbsp;|&nbsp; {wt}"
                )
            except Exception as exc:
                self.stl_status_html.value = f"<span style='color:#f88'>❌ Export failed: {exc}</span>"
            finally:
                self.stl_btn.disabled = False
                self.stl_progress.layout.visibility = "hidden"
                self.stl_progress.value = 0

        threading.Thread(target=_run, daemon=True).start()

    # ==================================================================
    # Status bar
    # ==================================================================

    def _status(self, msg: str) -> None:
        self.status_html.value = f"<span style='color:#aaa'>{msg}</span>"

    def _update_status(self) -> None:
        vol = self.volume
        if vol is None:
            return
        parts = [
            f"<b>Mode:</b> {vol.meta.get('mode','?')}",
            f"<b>Shape (t,y,x):</b> {'×'.join(str(d) for d in vol.shape)}",
        ]
        for key, label in [
            ("easing",        "Easing"),
            ("morphing",      "Morphing"),
            ("optical_flow",  "Opt.Flow"),
            ("edge_width",    "SDF σ"),
        ]:
            if key in vol.meta:
                parts.append(f"<b>{label}:</b> {vol.meta[key]}")
        sigma = vol.meta.get("smoothing_sigmas")
        if sigma and any(s > 0 for s in sigma):
            parts.append(f"<b>σ:</b> {sigma}")
        self.status_html.value = " &nbsp;|&nbsp; ".join(parts)

    def show(self):
        return self.ui
