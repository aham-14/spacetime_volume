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
from fastapi.staticfiles import StaticFiles
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
UPLOADED_VIDEO_PATH = OUTPUT_DIR / "uploaded_source.mp4"
INDEX_HTML_PATH = Path("static/index.html")

app = FastAPI(
    title="Space-Time Volume 4D-to-3D Engine",
    description="REST backend for 3D continuous scalar fields, SLA resin printing & wax/silicone casting.",
    version="4.5",
)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

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


def _generate_preview_glb(vol: SpaceTimeVolume) -> dict:
    """Extract and export realtime 3D GLB preview for <model-viewer>."""
    out_path = OUTPUT_DIR / "spacetime_model.glb"
    try:
        step = 2 if max(vol.shape) > 100 else 1
        exporter = MeshExporter(
            level=0.5,
            step_size=step,
            smooth_iterations=2,
            voxel_size=0.5,
        )
        stats = exporter.export(vol, out_path)
        timestamp = int(time.time() * 1000)
        return {
            "glb_url": f"/api/download/spacetime_model.glb?_t={timestamp}",
            "n_vertices": stats.n_vertices,
            "n_faces": stats.n_faces,
            "dimensions_mm": [round(float(d), 1) for d in stats.dimensions_mm],
            "volume_mm3": round(float(stats.volume_mm3), 1),
            "fits_standard_sla_vat": stats.fits_standard_sla_vat,
        }
    except Exception as exc:
        timestamp = int(time.time() * 1000)
        return {
            "glb_url": f"/api/download/spacetime_model.glb?_t={timestamp}",
            "error": str(exc),
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


class TrimRequest(BaseModel):
    start_frame: int = 0
    end_frame: int = 0  # 0 = all
    max_frames: int = 0  # 0 = all frames in trim
    max_dimension: int = 320
    use_flow: bool = True
    interp_factor: int = 2


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
    constant_boundary: Optional[str] = Field(None, description="Constant frame shape: None, rounded_rect, circle, box")
    boundary_mode: str = Field("framed", description="framed, monolith, or solid_relief")
    boundary_bezel: float = Field(4.0, ge=1.0, le=20.0)
    boundary_corner_radius: float = Field(8.0, ge=0.0, le=50.0)
    base_plate_slices: int = Field(0, ge=0, le=20)
    uninvert_y: bool = Field(True, description="Un-invert vertical Y axis so top of video is at top of 3D volume")
    invert_z: bool = Field(False, description="Invert time Z-axis direction")


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
        preview = _generate_preview_glb(vol)
        return {
            "status": "success",
            "shape": vol.shape,
            "meta": vol.meta,
            "intensity_range": [float(vol.data.min()), float(vol.data.max())],
            "video_url": "/api/video/source",
            "preview": preview,
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
        _sync_volume_to_video(vol, CURRENT_VIDEO_PATH)
        preview = _generate_preview_glb(vol)
        duration_s = round(time.time() - t0, 2)
        return {
            "status": "success",
            "shape": vol.shape,
            "n_frames": vol.n_frames,
            "duration_s": duration_s,
            "video_url": "/api/video/source",
            "preview": preview,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/upload/video")
async def upload_video(
    file: UploadFile = File(...),
    use_flow: bool = Form(True),
    interp_factor: int = Form(2),
    max_frames: int = Form(0),
    max_dimension: int = Form(320),
    start_frame: int = Form(0),
    end_frame: int = Form(0),
):
    try:
        t0 = time.time()
        content = await file.read()
        with open(UPLOADED_VIDEO_PATH, "wb") as f:
            f.write(content)

        probe_info = VideoIngestor.probe_video(UPLOADED_VIDEO_PATH)

        m_frames = max_frames if max_frames > 0 else None
        m_dim = max_dimension if max_dimension > 0 else None
        e_frame = end_frame if end_frame > 0 else None

        ingestor = VideoIngestor(
            max_frames=m_frames,
            max_dimension=m_dim,
            auto_stride=True,
            start_frame=start_frame,
            end_frame=e_frame,
        )
        raw_vol = ingestor.load(UPLOADED_VIDEO_PATH)

        if use_flow and raw_vol.n_frames >= 2:
            interpolator = OpticalFlowInterpolator(interp_factor=interp_factor, method="dis")
            vol = interpolator.interpolate(raw_vol)
        else:
            vol = raw_vol

        _SESSION["active_volume"] = vol
        preview = _generate_preview_glb(vol)
        duration_s = round(time.time() - t0, 2)
        return {
            "status": "success",
            "filename": file.filename,
            "shape": vol.shape,
            "n_frames": vol.n_frames,
            "duration_s": duration_s,
            "probe": probe_info,
            "video_url": "/api/video/source",
            "preview": preview,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to ingest video: {exc}")


@app.post("/api/video/trim")
def trim_video(req: TrimRequest):
    if not UPLOADED_VIDEO_PATH.exists():
        raise HTTPException(status_code=400, detail="No uploaded video found. Please upload an MP4 first.")
    try:
        t0 = time.time()
        m_frames = req.max_frames if req.max_frames > 0 else None
        m_dim = req.max_dimension if req.max_dimension > 0 else None
        e_frame = req.end_frame if req.end_frame > 0 else None

        ingestor = VideoIngestor(
            max_frames=m_frames,
            max_dimension=m_dim,
            auto_stride=True,
            start_frame=req.start_frame,
            end_frame=e_frame,
        )
        raw_vol = ingestor.load(UPLOADED_VIDEO_PATH)

        if req.use_flow and raw_vol.n_frames >= 2:
            interpolator = OpticalFlowInterpolator(interp_factor=req.interp_factor, method="dis")
            vol = interpolator.interpolate(raw_vol)
        else:
            vol = raw_vol

        _SESSION["active_volume"] = vol
        preview = _generate_preview_glb(vol)
        duration_s = round(time.time() - t0, 2)
        return {
            "status": "success",
            "shape": vol.shape,
            "n_frames": vol.n_frames,
            "duration_s": duration_s,
            "preview": preview,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to trim video: {exc}")


@app.get("/api/video/probe")
def probe_video_endpoint():
    if UPLOADED_VIDEO_PATH.exists():
        return VideoIngestor.probe_video(UPLOADED_VIDEO_PATH)
    return {"exists": False}


@app.get("/api/video/source")
def get_video_source():
    if UPLOADED_VIDEO_PATH.exists():
        return FileResponse(UPLOADED_VIDEO_PATH, media_type="video/mp4")
    if CURRENT_VIDEO_PATH.exists():
        return FileResponse(CURRENT_VIDEO_PATH, media_type="video/mp4")
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
            constant_boundary=req.constant_boundary,
            boundary_mode=req.boundary_mode,
            boundary_bezel=req.boundary_bezel,
            boundary_corner_radius=req.boundary_corner_radius,
            base_plate_slices=req.base_plate_slices,
            uninvert_y=req.uninvert_y,
            invert_z=req.invert_z,
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
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="static/index.html not found")
