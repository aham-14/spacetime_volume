# Space-Time Volume — Feature Implementation Checklist

Ordered from smallest effort to largest. Each item is committed individually.

---

| # | Status | Feature | Files Changed |
|---|---|---|---|
| F01 | [x] | Real `IntProgress` bar for STL export (replaces text spinner) | `viewer.py` |
| F02 | [ ] | OBJ / glTF export format selector (dropdown + `trimesh` auto-detect) | `mesh_export.py`, `viewer.py` |
| F03 | [ ] | `.npz` save / load on `SpaceTimeVolume` | `engine/volume.py`, `engine/__init__.py` |
| F04 | [ ] | `SpaceTimeVolume.crop()` and `.resize()` | `engine/volume.py`, `engine/__init__.py` |
| F05 | [ ] | Isosurface level slider in viewer (live STL param) | `viewer.py` |
| F06 | [ ] | Volume histogram widget in viewer | `engine/volume.py`, `viewer.py` |
| F07 | [ ] | Motion mask mode (`volume.motion_mask()` + UI toggle) | `engine/volume.py`, `viewer.py` |
| F08 | [ ] | Multi-keyframe procedural generator (N > 2 keyframes) | `engine/procedural.py`, `engine/__init__.py` |
| F09 | [ ] | DIS optical flow method (OpenCV built-in, zero extra deps) | `engine/optical_flow.py` |
| F10 | [ ] | 2D spatial tile occupancy index (skip empty XY blocks) | `engine/mesh_export.py` |
| F11 | [ ] | Composable `Pipeline` class | `engine/pipeline.py`, `engine/__init__.py` |
| F12 | [ ] | FastAPI REST backend (`api.py`) | `api.py`, `requirements.txt` |
| F13 | [ ] | 3D mesh preview via PyVista / trame | `engine/preview3d.py`, `engine/__init__.py` |
