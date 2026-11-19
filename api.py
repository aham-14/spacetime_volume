"""
api.py — Full-stack FastAPI backend and interactive 3D Web UI for Space-Time Volume.

Provides REST endpoints for volume generation, optical flow, video ingestion,
and mesh/mold export, plus an interactive dual-view Web Studio featuring Google
<model-viewer> and a synchronized Video & 2D Time-Slice Monitor.

Run with:
    python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from engine import (
    SpaceTimeVolume,
    ProceduralGenerator,
    VideoIngestor,
    OpticalFlowInterpolator,
    MeshExporter,
    EASING_NAMES,
    dummy_keyframes,
    write_dummy_mp4,
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_VIDEO_PATH = OUTPUT_DIR / "current_video.mp4"

app = FastAPI(
    title="Space-Time Volume 4D-to-3D Engine",
    description="REST backend for 3D continuous scalar fields, SLA resin printing & wax/silicone casting.",
    version="4.5",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache for active volume
_SESSION: Dict[str, Optional[SpaceTimeVolume]] = {
    "active_volume": None,
}


def _sync_volume_to_video(vol: SpaceTimeVolume, path: Path, fps: int = 24) -> None:
    """Encode the SpaceTimeVolume slices as an .mp4 video for browser playback."""
    try:
        T, H, W = vol.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(path), fourcc, float(fps), (W, H), isColor=False)
        for t in range(T):
            frame_u8 = (vol.frame(t) * 255.0).astype(np.uint8)
            vw.write(frame_u8)
        vw.release()
    except Exception:
        pass


def _get_active_volume() -> SpaceTimeVolume:
    vol = _SESSION["active_volume"]
    if vol is None:
        # Initialize default procedural demo and sync video
        fa, fb = dummy_keyframes()
        vol = ProceduralGenerator(fa, fb, edge_width=2.5).generate(24, "ease-in-out cubic")
        _SESSION["active_volume"] = vol
        if not CURRENT_VIDEO_PATH.exists():
            _sync_volume_to_video(vol, CURRENT_VIDEO_PATH)
    return vol


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class ProceduralRequest(BaseModel):
    n_frames: int = Field(24, ge=2, le=300)
    easing: str = Field("ease-in-out cubic")
    edge_width: float = Field(2.5, ge=0.5, le=10.0)


class ExportRequest(BaseModel):
    format: str = Field("glb", description="Export format: glb, stl, obj, ply")
    level: float = Field(0.5, ge=0.05, le=0.95)
    as_negative_mold: bool = False
    mold_open_top: bool = True
    mold_wall_thickness: int = Field(4, ge=2, le=16)
    as_hollow: bool = False
    hollow_wall_thickness: float = Field(3.0, ge=1.0, le=8.0)
    step_size: int = Field(1, ge=1, le=4)
    voxel_size_mm: float = Field(0.5, ge=0.05, le=5.0)
    smooth_iterations: int = Field(4, ge=0, le=20)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": "Space-Time Volume API",
        "version": "4.5",
        "easings": EASING_NAMES,
    }


@app.post("/api/generate/procedural")
def generate_procedural(req: ProceduralRequest):
    try:
        fa, fb = dummy_keyframes()
        gen = ProceduralGenerator(fa, fb, edge_width=req.edge_width)
        vol = gen.generate(n_frames=req.n_frames, transition_function=req.easing)
        _SESSION["active_volume"] = vol
        _sync_volume_to_video(vol, CURRENT_VIDEO_PATH)
        return {
            "status": "success",
            "shape": vol.shape,
            "meta": vol.meta,
            "intensity_range": [float(vol.data.min()), float(vol.data.max())],
            "video_url": "/api/video/source",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/generate/synthetic_video")
def generate_synthetic_video(use_flow: bool = True, interp_factor: int = 2):
    try:
        t0 = time.time()
        write_dummy_mp4(CURRENT_VIDEO_PATH, n=36)
        ingestor = VideoIngestor(max_frames=36, max_dimension=320)
        raw_vol = ingestor.load(CURRENT_VIDEO_PATH)

        if use_flow and raw_vol.n_frames >= 2:
            interpolator = OpticalFlowInterpolator(interp_factor=interp_factor, method="dis")
            vol = interpolator.interpolate(raw_vol)
        else:
            vol = raw_vol

        _SESSION["active_volume"] = vol
        duration_s = round(time.time() - t0, 2)
        return {
            "status": "success",
            "shape": vol.shape,
            "n_frames": vol.n_frames,
            "duration_s": duration_s,
            "video_url": "/api/video/source",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload/video")
async def upload_video(
    file: UploadFile = File(...),
    use_flow: bool = Form(True),
    interp_factor: int = Form(2),
    max_frames: int = Form(48),
    max_dimension: int = Form(320),
):
    try:
        t0 = time.time()
        content = await file.read()
        with open(CURRENT_VIDEO_PATH, "wb") as f:
            f.write(content)

        m_frames = max_frames if max_frames > 0 else None
        m_dim = max_dimension if max_dimension > 0 else None

        ingestor = VideoIngestor(
            max_frames=m_frames,
            max_dimension=m_dim,
            auto_stride=True,
        )
        raw_vol = ingestor.load(CURRENT_VIDEO_PATH)

        if use_flow and raw_vol.n_frames >= 2:
            # Use DIS optical flow which is 5x faster than Farneback
            interpolator = OpticalFlowInterpolator(interp_factor=interp_factor, method="dis")
            vol = interpolator.interpolate(raw_vol)
        else:
            vol = raw_vol

        _SESSION["active_volume"] = vol
        duration_s = round(time.time() - t0, 2)
        return {
            "status": "success",
            "filename": file.filename,
            "shape": vol.shape,
            "n_frames": vol.n_frames,
            "duration_s": duration_s,
            "video_url": "/api/video/source",
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to ingest video: {exc}")


@app.get("/api/video/source")
def get_video_source():
    if not CURRENT_VIDEO_PATH.exists():
        vol = _get_active_volume()
        _sync_volume_to_video(vol, CURRENT_VIDEO_PATH)
    return FileResponse(CURRENT_VIDEO_PATH, media_type="video/mp4")


@app.get("/api/video/frame/{t}")
def get_video_frame(t: int):
    vol = _get_active_volume()
    t_idx = int(np.clip(t, 0, vol.n_frames - 1))
    frame_arr = (vol.frame(t_idx) * 255.0).astype(np.uint8)

    # Convert to 3-channel for browser image rendering
    bgr = cv2.cvtColor(frame_arr, cv2.COLOR_GRAY2BGR)
    # Add small visual time stamp
    cv2.putText(
        bgr,
        f"Z-Time: {t_idx} / {vol.n_frames - 1}",
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 120),
        1,
        cv2.LINE_AA,
    )

    success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/video/info")
def get_video_info():
    vol = _get_active_volume()
    return {
        "n_frames": vol.n_frames,
        "shape": vol.shape,
        "has_video": CURRENT_VIDEO_PATH.exists(),
    }


@app.post("/api/export")
def export_mesh(req: ExportRequest):
    vol = _get_active_volume()

    fmt = req.format.lower().lstrip(".")
    if fmt not in ("glb", "stl", "obj", "ply"):
        raise HTTPException(status_code=400, detail=f"Unsupported format '{req.format}'")

    target_prefix = "mold" if req.as_negative_mold else ("hollow" if req.as_hollow else "model")
    filename = f"spacetime_{target_prefix}.{fmt}"
    out_path = OUTPUT_DIR / filename

    try:
        exporter = MeshExporter(
            level=req.level,
            step_size=req.step_size,
            smooth_iterations=req.smooth_iterations,
            voxel_size=req.voxel_size_mm,
            as_negative_mold=req.as_negative_mold,
            mold_wall_thickness=req.mold_wall_thickness,
            mold_open_top=req.mold_open_top,
            as_hollow=req.as_hollow,
            hollow_wall_thickness=req.hollow_wall_thickness,
        )
        stats = exporter.export(vol, out_path, level=req.level)

        return {
            "status": "success",
            "filename": filename,
            "download_url": f"/api/download/{filename}",
            "stats": {
                "n_vertices": stats.n_vertices,
                "n_faces": stats.n_faces,
                "volume_mm3": stats.volume_mm3,
                "is_watertight": stats.is_watertight,
                "dimensions_mm": stats.dimensions_mm,
                "fits_standard_sla_vat": stats.fits_standard_sla_vat,
                "format": stats.export_format,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/download/{filename}")
def download_mesh(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Mesh file not found")
    media_type = "model/gltf-binary" if filename.endswith(".glb") else "application/octet-stream"
    return FileResponse(file_path, filename=filename, media_type=media_type)


# ---------------------------------------------------------------------------
# Interactive Web Client with Google <model-viewer> & Video Monitor
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index_page():
    easings_options = "\n".join([f'<option value="{name}">{name}</option>' for name in EASING_NAMES])
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Space-Time Volume — 4D to 3D Print Studio</title>
  <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f111a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; overflow: hidden;
    }}
    header {{
      background: #181c2b; padding: 12px 24px; border-bottom: 1px solid #2d3748;
      display: flex; justify-content: space-between; align-items: center; z-index: 10;
    }}
    header .title-group {{ display: flex; align-items: center; gap: 12px; }}
    header h1 {{ font-size: 1.25rem; font-weight: 700; color: #60a5fa; }}
    header .badge {{ background: #2563eb; color: white; padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; }}
    .header-actions {{ display: flex; gap: 10px; align-items: center; }}
    .pill-btn {{
      background: #334155; color: white; border: 1px solid #475569; padding: 6px 14px;
      border-radius: 6px; font-size: 0.8rem; cursor: pointer; display: flex; align-items: center; gap: 6px;
    }}
    .pill-btn:hover {{ background: #475569; }}
    .pill-btn.active {{ background: #2563eb; border-color: #3b82f6; }}

    main {{
      display: flex; flex: 1; overflow: hidden; position: relative;
    }}
    .sidebar {{
      width: 360px; background: #131722; border-right: 1px solid #2d3748; padding: 16px;
      overflow-y: auto; display: flex; flex-direction: column; gap: 14px; z-index: 5;
    }}
    .panel {{
      background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 14px;
    }}
    .panel h2 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 10px; display: flex; justify-content: space-between; }}
    .field {{ margin-bottom: 10px; display: flex; flex-direction: column; gap: 4px; }}
    label {{ font-size: 0.8rem; color: #cbd5e1; }}
    select, input[type="number"], input[type="range"], input[type="text"] {{
      background: #0f111a; border: 1px solid #475569; color: white; padding: 6px 10px; border-radius: 4px; font-size: 0.85rem;
    }}
    .checkbox-field {{ display: flex; align-items: center; gap: 8px; font-size: 0.85rem; margin-bottom: 8px; }}
    button.btn-primary {{
      background: #2563eb; color: white; border: none; padding: 9px 16px; border-radius: 6px;
      font-weight: 600; cursor: pointer; transition: background 0.15s; width: 100%;
    }}
    button.btn-primary:hover {{ background: #1d4ed8; }}
    button.btn-secondary {{
      background: #334155; color: #f1f5f9; border: 1px solid #475569; padding: 7px 12px; border-radius: 6px;
      font-weight: 500; font-size: 0.8rem; cursor: pointer; width: 100%; margin-top: 4px;
    }}
    button.btn-secondary:hover {{ background: #475569; }}

    .workspace {{
      flex: 1; display: flex; position: relative; background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
    }}
    .viewport-3d {{
      flex: 1; position: relative; width: 100%; height: 100%;
    }}
    model-viewer {{
      width: 100%; height: 100%;
    }}

    /* Video & 2D Frame Monitor Dock */
    .video-monitor-dock {{
      width: 380px; background: rgba(19, 23, 34, 0.95); border-left: 1px solid #2d3748;
      display: flex; flex-direction: column; backdrop-filter: blur(8px); z-index: 8;
      transition: width 0.2s ease, transform 0.2s ease;
    }}
    .video-monitor-dock.hidden {{
      display: none;
    }}
    .dock-header {{
      padding: 12px 16px; background: #181c2b; border-bottom: 1px solid #2d3748;
      display: flex; justify-content: space-between; align-items: center;
    }}
    .dock-header h3 {{ font-size: 0.9rem; color: #60a5fa; }}
    .dock-tabs {{ display: flex; gap: 4px; }}
    .dock-tab {{
      background: #0f111a; border: 1px solid #334155; color: #94a3b8; padding: 4px 10px;
      border-radius: 4px; font-size: 0.75rem; cursor: pointer;
    }}
    .dock-tab.active {{ background: #2563eb; color: white; border-color: #3b82f6; }}

    .dock-content {{
      flex: 1; padding: 14px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
    }}
    .video-screen-box {{
      width: 100%; height: 240px; background: #000; border: 1px solid #334155; border-radius: 6px;
      display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden;
    }}
    .video-screen-box video, .video-screen-box img {{
      max-width: 100%; max-height: 100%; object-fit: contain;
    }}
    .scrubber-controls {{
      display: flex; flex-direction: column; gap: 8px; background: #1a202c; padding: 12px; border-radius: 6px; border: 1px solid #2d3748;
    }}
    .scrubber-row {{ display: flex; align-items: center; gap: 10px; }}
    .btn-play {{
      background: #10b981; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;
    }}
    .btn-play.playing {{ background: #ef4444; }}

    .overlay {{
      position: absolute; bottom: 20px; left: 20px; background: rgba(15, 23, 42, 0.85);
      border: 1px solid #334155; padding: 10px 16px; border-radius: 8px; font-size: 0.8rem;
      backdrop-filter: blur(4px); display: flex; gap: 16px; z-index: 6;
    }}
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>⏳ Space-Time Volume Studio</h1>
      <span class="badge">4D → 3D SLA & Casting</span>
    </div>
    <div class="header-actions">
      <button class="pill-btn active" id="btn_toggle_monitor">
        📹 Video & Frame Monitor
      </button>
    </div>
  </header>
  <main>
    <div class="sidebar">
      <!-- Section A: Mode Selection / Generation -->
      <div class="panel">
        <h2>1. Ingestion / Generation Mode</h2>
        <div class="field">
          <label>Mode:</label>
          <select id="mode_select">
            <option value="procedural" selected>⚡ Mode 2 — SDF Procedural Morphing</option>
            <option value="video">🎥 Mode 1 — Video Ingestion & Optical Flow</option>
          </select>
        </div>

        <div id="controls_procedural">
          <div class="field">
            <label>Time Depth (Frames):</label>
            <input type="number" id="n_frames" value="24" min="2" max="120">
          </div>
          <div class="field">
            <label>Easing Curve:</label>
            <select id="easing">
              {easings_options}
            </select>
          </div>
          <div class="field">
            <label>SDF Edge Width (px):</label>
            <input type="number" id="edge_width" value="2.5" step="0.5" min="0.5" max="8.0">
          </div>
          <button class="btn-primary" id="btn_generate_proc">Generate Procedural Volume</button>
        </div>

        <div id="controls_video" style="display:none;">
          <div class="checkbox-field">
            <input type="checkbox" id="video_use_flow" checked>
            <label for="video_use_flow">DIS Optical Flow (2× Sub-frames)</label>
          </div>
          <div class="field" style="margin-top:6px;">
            <label>Slices across Video:</label>
            <select id="video_max_frames">
              <option value="36">36 slices (Blazing Fast)</option>
              <option value="48" selected>48 slices (Recommended)</option>
              <option value="72">72 slices (High Detail)</option>
              <option value="120">120 slices (Fine Layer)</option>
              <option value="0">All Frames (Full Video)</option>
            </select>
          </div>
          <div class="field">
            <label>Processing Resolution:</label>
            <select id="video_max_dimension">
              <option value="320" selected>320 px (Fast Instant 3D)</option>
              <option value="480">480 px (Balanced Detail)</option>
              <option value="0">Original Resolution</option>
            </select>
          </div>
          <button class="btn-secondary" id="btn_synthetic_video">🎥 Load Orbiting Synthetic Video</button>
          <div style="margin-top: 8px;">
            <label style="display:block; margin-bottom:4px; font-weight:bold;">Upload MP4 / MOV Video:</label>
            <input type="file" id="video_file_input" accept="video/mp4,video/*" style="font-size:0.75rem; width:100%;">
          </div>
        </div>
      </div>

      <!-- Section B: 3D Hardware Export -->
      <div class="panel">
        <h2>2. Hardware 3D Export</h2>
        <div class="field">
          <label>Format:</label>
          <select id="format">
            <option value="glb" selected>GLB (.glb) — Realtime 3D</option>
            <option value="stl">STL (.stl) — Slicer</option>
            <option value="obj">OBJ (.obj) — Blender</option>
            <option value="ply">PLY (.ply) — Point/Mesh</option>
          </select>
        </div>
        <div class="field">
          <label>Isosurface Level:</label>
          <input type="range" id="level" min="0.1" max="0.9" step="0.05" value="0.5">
          <span id="level_val" style="font-size:0.75rem; color:#94a3b8;">0.50</span>
        </div>
        <div class="checkbox-field">
          <input type="checkbox" id="as_hollow">
          <label for="as_hollow">🏺 Hollow Shell (Resin Saving + Drain)</label>
        </div>
        <div class="checkbox-field">
          <input type="checkbox" id="as_mold">
          <label for="as_mold">📦 Negative Pouring Mold</label>
        </div>
        <button class="btn-primary" id="btn_export">🖨 Export 3D Mesh</button>
      </div>

      <!-- Section C: Status -->
      <div class="panel" id="status_panel">
        <h2>Status</h2>
        <p id="status_msg" style="font-size:0.8rem; color:#94a3b8;">Ready.</p>
      </div>
    </div>

    <!-- Central Workspace: 3D Viewport + Right Dock Video Monitor -->
    <div class="workspace">
      <div class="viewport-3d">
        <model-viewer id="viewer3d" src="/api/download/spacetime_model.glb" camera-controls auto-rotate shadow-intensity="1" ar></model-viewer>
        <div class="overlay" id="stats_overlay">
          <span id="stat_verts">Verts: 14,818</span>
          <span id="stat_faces">Faces: 29,632</span>
          <span id="stat_dims">Box: 46.5 × 32.5 × 12.0 mm</span>
          <span id="stat_sla">SLA Vat: ✅</span>
        </div>
      </div>

      <!-- Right Dock: Video & 2D Frame Time-Slice Stream -->
      <div class="video-monitor-dock" id="video_monitor">
        <div class="dock-header">
          <h3>📹 Video & Slice Stream</h3>
          <div class="dock-tabs">
            <button class="dock-tab active" id="tab_slice">2D Z-Slice</button>
            <button class="dock-tab" id="tab_video">MP4 Video</button>
          </div>
        </div>

        <div class="dock-content">
          <!-- Screen Box -->
          <div class="video-screen-box">
            <img id="slice_screen" src="/api/video/frame/0" alt="2D Time Slice">
            <video id="video_screen" src="/api/video/source" controls loop autoplay style="display:none;"></video>
          </div>

          <!-- Time Scrubber Controls -->
          <div class="scrubber-controls" id="scrubber_panel">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem;">
              <span id="lbl_frame_idx">Time (Z): Frame 0 / 23</span>
              <span id="lbl_z_height" style="color:#60a5fa;">Z = 0.0 mm</span>
            </div>
            <div class="scrubber-row">
              <button class="btn-play" id="btn_play">▶</button>
              <input type="range" id="time_slider" min="0" max="23" value="0" style="flex:1;">
            </div>
          </div>

          <div style="font-size:0.75rem; color:#64748b; line-height:1.4;">
            <b>About this Monitor:</b><br>
            As frames advance in time along the Z-axis, they are stacked continuously into the 3D scalar field. Watch the 2D slices morph in real time, or scrub manually to inspect any cross-section.
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    // Elements
    const modeSelect = document.getElementById('mode_select');
    const controlsProc = document.getElementById('controls_procedural');
    const controlsVid = document.getElementById('controls_video');
    const statusMsg = document.getElementById('status_msg');
    const viewer = document.getElementById('viewer3d');
    const overlay = document.getElementById('stats_overlay');

    const videoMonitor = document.getElementById('video_monitor');
    const btnToggleMonitor = document.getElementById('btn_toggle_monitor');
    const tabSlice = document.getElementById('tab_slice');
    const tabVideo = document.getElementById('tab_video');
    const sliceScreen = document.getElementById('slice_screen');
    const videoScreen = document.getElementById('video_screen');
    const scrubberPanel = document.getElementById('scrubber_panel');

    const timeSlider = document.getElementById('time_slider');
    const btnPlay = document.getElementById('btn_play');
    const lblFrameIdx = document.getElementById('lbl_frame_idx');
    const lblZHeight = document.getElementById('lbl_z_height');

    const levelSlider = document.getElementById('level');
    const levelVal = document.getElementById('level_val');
    levelSlider.oninput = () => levelVal.innerText = parseFloat(levelSlider.value).toFixed(2);

    let totalFrames = 24;
    let isPlaying = false;
    let playTimer = null;

    // Toggle Monitor Panel
    btnToggleMonitor.onclick = () => {{
      videoMonitor.classList.toggle('hidden');
      btnToggleMonitor.classList.toggle('active');
    }};

    // Switch Monitor Tabs (2D Slice vs Video Player)
    tabSlice.onclick = () => {{
      tabSlice.classList.add('active');
      tabVideo.classList.remove('active');
      sliceScreen.style.display = 'block';
      videoScreen.style.display = 'none';
      scrubberPanel.style.display = 'flex';
    }};
    tabVideo.onclick = () => {{
      tabVideo.classList.add('active');
      tabSlice.classList.remove('active');
      videoScreen.style.display = 'block';
      sliceScreen.style.display = 'none';
      scrubberPanel.style.display = 'none';
      videoScreen.play();
    }};

    // Mode Selector
    modeSelect.onchange = () => {{
      if (modeSelect.value === 'procedural') {{
        controlsProc.style.display = 'block';
        controlsVid.style.display = 'none';
      }} else {{
        controlsProc.style.display = 'none';
        controlsVid.style.display = 'block';
      }}
    }};

    // Time Scrubber
    function updateFrame(t) {{
      timeSlider.value = t;
      sliceScreen.src = `/api/video/frame/${{t}}?_t=${{Date.now()}}`;
      lblFrameIdx.innerText = `Time (Z): Frame ${{t}} / ${{totalFrames - 1}}`;
      lblZHeight.innerText = `Z = ${{(t * 0.5).toFixed(1)}} mm`;
    }}

    timeSlider.oninput = (e) => {{
      if (isPlaying) stopPlayback();
      updateFrame(parseInt(e.target.value));
    }};

    function startPlayback() {{
      isPlaying = true;
      btnPlay.innerText = "⏸";
      btnPlay.classList.add('playing');
      playTimer = setInterval(() => {{
        let nextT = (parseInt(timeSlider.value) + 1) % totalFrames;
        updateFrame(nextT);
      }}, 65); // ~15-20 fps
    }}

    function stopPlayback() {{
      isPlaying = false;
      btnPlay.innerText = "▶";
      btnPlay.classList.remove('playing');
      if (playTimer) clearInterval(playTimer);
      playTimer = null;
    }}

    btnPlay.onclick = () => {{
      if (isPlaying) stopPlayback();
      else startPlayback();
    }};

    // Procedural Volume Generation
    document.getElementById('btn_generate_proc').onclick = async () => {{
      statusMsg.innerText = "Generating procedural volume...";
      const res = await fetch('/api/generate/procedural', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          n_frames: parseInt(document.getElementById('n_frames').value),
          easing: document.getElementById('easing').value,
          edge_width: parseFloat(document.getElementById('edge_width').value)
        }})
      }});
      const data = await res.json();
      totalFrames = data.shape[0];
      timeSlider.max = totalFrames - 1;
      updateFrame(0);
      videoScreen.src = `/api/video/source?_t=${{Date.now()}}`;
      statusMsg.innerText = `Volume ready: ${{data.shape.join(' × ')}} voxels`;
    }};

    // Synthetic Video Load
    document.getElementById('btn_synthetic_video').onclick = async () => {{
      statusMsg.innerHTML = "<span style='color:#60a5fa;'>⏳ Generating synthetic video & DIS optical flow...</span>";
      try {{
        const res = await fetch('/api/generate/synthetic_video', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            use_flow: document.getElementById('video_use_flow').checked,
            interp_factor: 2
          }})
        }});
        if (!res.ok) {{
          const err = await res.json();
          statusMsg.innerHTML = `<span style="color:#f87171;">❌ Generation failed: ${{err.detail || 'Server error'}}</span>`;
          return;
        }}
        const data = await res.json();
        totalFrames = data.shape[0];
        timeSlider.max = totalFrames - 1;
        updateFrame(0);
        videoScreen.src = `/api/video/source?_t=${{Date.now()}}`;
        statusMsg.innerHTML = `<span style="color:#4ade80;">✔ Synthetic video volume ready: ${{data.shape.join(' × ')}} (${{data.duration_s || 0.1}}s)</span>`;
      }} catch (err) {{
        statusMsg.innerHTML = `<span style="color:#f87171;">❌ Error: ${{err.message}}</span>`;
      }}
    }};

    // Video File Upload
    document.getElementById('video_file_input').onchange = async (e) => {{
      const file = e.target.files[0];
      if (!file) return;
      statusMsg.innerHTML = `<span style="color:#60a5fa;">⏳ Ingesting ${{file.name}} & processing optical flow...</span>`;
      try {{
        const formData = new FormData();
        formData.append('file', file);
        formData.append('use_flow', document.getElementById('video_use_flow').checked);
        formData.append('interp_factor', 2);
        formData.append('max_frames', parseInt(document.getElementById('video_max_frames').value));
        formData.append('max_dimension', parseInt(document.getElementById('video_max_dimension').value));

        const res = await fetch('/api/upload/video', {{
          method: 'POST',
          body: formData
        }});
        if (!res.ok) {{
          const err = await res.json();
          statusMsg.innerHTML = `<span style="color:#f87171;">❌ Ingestion failed: ${{err.detail || 'Server error'}}</span>`;
          return;
        }}
        const data = await res.json();
        totalFrames = data.shape[0];
        timeSlider.max = totalFrames - 1;
        updateFrame(0);
        videoScreen.src = `/api/video/source?_t=${{Date.now()}}`;
        statusMsg.innerHTML = `<span style="color:#4ade80;">✔ Ingested ${{data.filename}} (${{data.shape.join(' × ')}} in ${{data.duration_s}}s)</span>`;
      }} catch (err) {{
        statusMsg.innerHTML = `<span style="color:#f87171;">❌ Upload error: ${{err.message}}</span>`;
      }}
    }};

    // 3D Mesh Export
    document.getElementById('btn_export').onclick = async () => {{
      statusMsg.innerHTML = "<span style='color:#60a5fa;'>⏳ Extracting & smoothing 3D mesh...</span>";
      const fmt = document.getElementById('format').value;
      const bShape = document.getElementById('constant_boundary').value;
      try {{
        const res = await fetch('/api/export', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            format: fmt,
            level: parseFloat(levelSlider.value),
            as_hollow: document.getElementById('as_hollow').checked,
            as_negative_mold: document.getElementById('as_mold').checked,
            constant_boundary: bShape !== 'None' ? bShape : null,
            boundary_mode: document.getElementById('boundary_mode').value,
            boundary_bezel: parseFloat(document.getElementById('boundary_bezel').value),
            base_plate_slices: parseInt(document.getElementById('base_plate_slices').value)
          }})
        }});
        if (!res.ok) {{
          const err = await res.json();
          statusMsg.innerHTML = `<span style="color:#f87171;">❌ Export failed: ${{err.detail || 'Server error'}}</span>`;
          return;
        }}
        const data = await res.json();
        if (data.status === 'success') {{
          statusMsg.innerHTML = `Exported <a href="${{data.download_url}}" download style="color:#60a5fa; font-weight:bold;">Download ${{data.filename}}</a>`;
          if (fmt === 'glb') {{
            viewer.src = data.download_url;
          }}
          overlay.style.display = 'flex';
          document.getElementById('stat_verts').innerText = `Verts: ${{data.stats.n_vertices.toLocaleString()}}`;
          document.getElementById('stat_faces').innerText = `Faces: ${{data.stats.n_faces.toLocaleString()}}`;
          const d = data.stats.dimensions_mm;
          document.getElementById('stat_dims').innerText = `Box: ${{d[0].toFixed(1)}} × ${{d[1].toFixed(1)}} × ${{d[2].toFixed(1)}} mm`;
          document.getElementById('stat_sla').innerText = data.stats.fits_standard_sla_vat ? "SLA Vat: ✅" : "SLA Vat: ⚠ Exceeds";
        }} else {{
          statusMsg.innerText = "Export failed.";
        }}
      }} catch (err) {{
        statusMsg.innerHTML = `<span style="color:#f87171;">❌ Export error: ${{err.message}}</span>`;
      }}
    }};
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
