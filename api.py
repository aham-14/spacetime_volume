"""
api.py — Full-stack FastAPI backend and interactive 3D Web UI for Space-Time Volume.

Provides REST endpoints for volume generation, optical flow, and mesh/mold export,
plus a self-contained web client using Google <model-viewer> for real-time 3D inspection.

Run with:
    python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

import numpy as np

from engine import (
    SpaceTimeVolume,
    ProceduralGenerator,
    VideoIngestor,
    OpticalFlowInterpolator,
    MeshExporter,
    EASING_NAMES,
    dummy_keyframes,
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Space-Time Volume 4D-to-3D Engine",
    description="REST backend for 3D continuous scalar fields, SLA resin printing & wax/silicone casting.",
    version="4.0",
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


def _get_active_volume() -> SpaceTimeVolume:
    vol = _SESSION["active_volume"]
    if vol is None:
        # Initialize default procedural demo
        fa, fb = dummy_keyframes()
        vol = ProceduralGenerator(fa, fb, edge_width=2.5).generate(24, "ease-in-out cubic")
        _SESSION["active_volume"] = vol
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
        "version": "4.0",
        "easings": EASING_NAMES,
    }


@app.post("/api/generate/procedural")
def generate_procedural(req: ProceduralRequest):
    try:
        fa, fb = dummy_keyframes()
        gen = ProceduralGenerator(fa, fb, edge_width=req.edge_width)
        vol = gen.generate(n_frames=req.n_frames, transition_function=req.easing)
        _SESSION["active_volume"] = vol
        return {
            "status": "success",
            "shape": vol.shape,
            "meta": vol.meta,
            "intensity_range": [float(vol.data.min()), float(vol.data.max())],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
# Interactive Web Client with Google <model-viewer>
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
      background: #0f111a; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column;
    }}
    header {{
      background: #181c2b; padding: 14px 24px; border-bottom: 1px solid #2d3748;
      display: flex; justify-content: space-between; align-items: center;
    }}
    header h1 {{ font-size: 1.25rem; font-weight: 700; color: #60a5fa; }}
    header .badge {{ background: #2563eb; color: white; padding: 3px 10px; border-radius: 9999px; font-size: 0.75rem; }}
    main {{
      display: flex; flex: 1; overflow: hidden;
    }}
    .sidebar {{
      width: 360px; background: #131722; border-right: 1px solid #2d3748; padding: 20px;
      overflow-y: auto; display: flex; flex-direction: column; gap: 16px;
    }}
    .panel {{
      background: #1a202c; border: 1px solid #2d3748; border-radius: 8px; padding: 14px;
    }}
    .panel h2 {{ font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 12px; }}
    .field {{ margin-bottom: 12px; display: flex; flex-direction: column; gap: 4px; }}
    label {{ font-size: 0.8rem; color: #cbd5e1; }}
    select, input[type="number"], input[type="range"] {{
      background: #0f111a; border: 1px solid #475569; color: white; padding: 6px 10px; border-radius: 4px; font-size: 0.85rem;
    }}
    .checkbox-field {{ display: flex; align-items: center; gap: 8px; font-size: 0.85rem; margin-bottom: 8px; }}
    button {{
      background: #2563eb; color: white; border: none; padding: 10px 16px; border-radius: 6px;
      font-weight: 600; cursor: pointer; transition: background 0.15s; width: 100%;
    }}
    button:hover {{ background: #1d4ed8; }}
    button:disabled {{ background: #475569; cursor: not-allowed; }}
    .viewport {{
      flex: 1; position: relative; background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
      display: flex; flex-direction: column; align-items: center; justify-content: center;
    }}
    model-viewer {{
      width: 100%; height: 100%;
    }}
    .overlay {{
      position: absolute; bottom: 20px; background: rgba(15, 23, 42, 0.85);
      border: 1px solid #334155; padding: 10px 16px; border-radius: 8px; font-size: 0.8rem;
      backdrop-filter: blur(4px); display: flex; gap: 16px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>⏳ Space-Time Volume Studio</h1>
    <span class="badge">SLA & Casting 4.0</span>
  </header>
  <main>
    <div class="sidebar">
      <div class="panel">
        <h2>1. Procedural Generation</h2>
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
        <button id="btn_generate">Generate Volume</button>
      </div>

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
          <label for="as_hollow">🏺 Hollow Shell (Resin Saving)</label>
        </div>
        <div class="checkbox-field">
          <input type="checkbox" id="as_mold">
          <label for="as_mold">📦 Negative Pouring Mold</label>
        </div>
        <button id="btn_export">🖨 Export 3D Mesh</button>
      </div>

      <div class="panel" id="status_panel">
        <h2>Status</h2>
        <p id="status_msg" style="font-size:0.8rem; color:#94a3b8;">Ready.</p>
      </div>
    </div>

    <div class="viewport">
      <model-viewer id="viewer3d" src="/api/download/spacetime_model.glb" camera-controls auto-rotate shadow-intensity="1" ar></model-viewer>
      <div class="overlay" id="stats_overlay">
        <span id="stat_verts">Verts: 14,818</span>
        <span id="stat_faces">Faces: 29,632</span>
        <span id="stat_dims">Box: 46.5 × 32.5 × 12.0 mm</span>
        <span id="stat_sla">SLA Vat: ✅</span>
      </div>
    </div>
  </main>

  <script>
    const levelSlider = document.getElementById('level');
    const levelVal = document.getElementById('level_val');
    levelSlider.oninput = () => levelVal.innerText = parseFloat(levelSlider.value).toFixed(2);

    const statusMsg = document.getElementById('status_msg');
    const viewer = document.getElementById('viewer3d');
    const overlay = document.getElementById('stats_overlay');

    document.getElementById('btn_generate').onclick = async () => {{
      statusMsg.innerText = "Generating volume...";
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
      statusMsg.innerText = "Volume ready: " + data.shape.join(' × ');
    }};

    document.getElementById('btn_export').onclick = async () => {{
      statusMsg.innerText = "Extracting & smoothing 3D mesh...";
      const fmt = document.getElementById('format').value;
      const res = await fetch('/api/export', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{
          format: fmt,
          level: parseFloat(levelSlider.value),
          as_hollow: document.getElementById('as_hollow').checked,
          as_negative_mold: document.getElementById('as_mold').checked
        }})
      }});
      const data = await res.json();
      if (data.status === 'success') {{
        statusMsg.innerHTML = `Exported <a href="${{data.download_url}}" download style="color:#60a5fa">Download ${{data.filename}}</a>`;
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
    }};
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
