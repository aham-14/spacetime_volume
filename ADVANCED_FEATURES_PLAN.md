# Advanced Space-Time Volume Features Plan

This plan outlines the next phase of professional SLA resin manufacturing, casting, mesh repair, and full-stack web integration.

---

## 📋 Task Breakdown

### Task 6: SLA Resin Printability — Uniform Hollowing & Drain Holes
- [x] **6.1**: Implement `hollow()` in `engine/volume.py` using distance transform / SDF offset
- [x] **6.2**: Add drain hole puncher through base slices for SLA resin peel suction release
- [x] **6.3**: Wire hollowing parameters into `MeshExporter` and `SpaceTimeViewer` export panel
- [x] **6.4**: Unit tests in `tests/test_hollowing.py` (passed)

### Task 7: 2-Piece Mold Splitting with Registration Pegs & Sockets
- [x] **7.1**: Implement `generate_two_piece_mold()` in `engine/mesh_export.py`
- [x] **7.2**: Carve interlocking registration keys (raised pegs on Part A, negative sockets on Part B)
- [x] **7.3**: Expose 2-piece mold export in `MeshExporter` and `SpaceTimeViewer`
- [x] **7.4**: Unit tests in `tests/test_two_piece_mold.py` (passed)

### Task 8: Pre-Flight Slicer Health Check & Physical Vat Dimensions
- [x] **8.1**: Integrate automated mesh healing in `MeshExporter` (`trimesh.repair.fix_normals`, `fill_holes`)
- [x] **8.2**: Compute physical dimensions ($W \times D \times H$ mm) and SLA build volume compatibility in `MeshStats`
- [x] **8.3**: Display dimensions and vat fit warnings in `viewer.py` status box
- [x] **8.4**: Unit tests in `tests/test_slicer_health.py` (passed)

### Task 9: Full-Stack Web App (FastAPI + Three.js / `<model-viewer>` UI)
- [x] **9.1**: Create `api.py` with FastAPI endpoints (`/health`, `/api/generate/procedural`, `/api/export`, `/api/download`, `/`)
- [x] **9.2**: Update `requirements.txt` with `fastapi`, `uvicorn`, `python-multipart`
- [x] **9.3**: Integration tests for FastAPI endpoints in `tests/test_api.py` (passed)

---
