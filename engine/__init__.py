"""
engine — Space-Time Volume backend.

Pure scientific and graphics backend for Space-Time Volume generation,
SDF interpolation, optical flow processing, and SLA/casting mesh generation.
"""
from .volume import SpaceTimeVolume
from .video_ingest import VideoIngestor
from .optical_flow import OpticalFlowInterpolator
from .procedural import ProceduralGenerator
from .easing import EASINGS, EASING_NAMES, get_easing, EasingFn
from .sdf_shapes import (
    sdf_disc, sdf_rect, sdf_ring, sdf_capsule,
    sdf_smooth_union, sdf_to_intensity, make_grids,
)
from .mesh_export import (
    MeshExporter,
    MeshStats,
    chunked_marching_cubes,
    generate_negative_mold,
    SparseOccupancyIndex,
)
from .pipeline import Pipeline, StepTelemetry
from .dummy_data import dummy_keyframes, dummy_video_array, write_dummy_mp4

__all__ = [
    # Core data
    "SpaceTimeVolume",
    # Mode 1
    "VideoIngestor",
    "OpticalFlowInterpolator",
    # Mode 2
    "ProceduralGenerator",
    # Easing curves
    "EASINGS", "EASING_NAMES", "get_easing", "EasingFn",
    # SDF primitives
    "sdf_disc", "sdf_rect", "sdf_ring", "sdf_capsule",
    "sdf_smooth_union", "sdf_to_intensity", "make_grids",
    # Mesh export & Mold generation
    "MeshExporter", "MeshStats", "chunked_marching_cubes",
    "generate_negative_mold", "SparseOccupancyIndex",
    # Composable pipeline
    "Pipeline", "StepTelemetry",
    # Dummy data
    "dummy_keyframes", "dummy_video_array", "write_dummy_mp4",
]
