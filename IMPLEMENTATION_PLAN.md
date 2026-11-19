# Space-Time Volume — Feature Implementation Plan

Track progress and implementation status for new enhancements to the Space-Time Volume engine and viewer.

---

## 📋 Feature Tasks

- [x] **Task 1: SpaceTimeVolume Data Container Enhancements**
  - [x] Add `crop(t0, t1, y0, y1, x0, x1)`: Extract 3D sub-volume / ROI
  - [x] Add `resize(new_shape)`: Subsample or supersample volume using `scipy.ndimage.zoom`
  - [x] Add `save(path)` and `load(path)`: Compressed `.npz` storage with serialized metadata
  - [x] Add `histogram(bins, range)`: Compute voxel intensity distributions
  - [x] Add `motion_mask(threshold)`: Temporal difference field isolating dynamic features
  - [x] Unit tests in test suite (5 passed)

- [x] **Task 2: Multi-Format Mesh Export (STL, OBJ, PLY, GLB) & Isosurface Level Control**
  - [x] Update `MeshExporter.export()` to accept `level` override and support `.stl`, `.obj`, `.ply`, `.glb`
  - [x] Update `app.py` `mesh_export_factory` to wire dynamic `level`
  - [x] Add `level_slider` (range 0.05–0.95, step 0.05) to `SpaceTimeViewer` export panel
  - [x] Add `format_dropdown` (STL, OBJ, PLY, GLB) in `SpaceTimeViewer` with auto path updating
  - [x] Unit tests for multi-format export and variable isosurface levels (passed)

- [x] **Task 3: Fast DIS Optical Flow Option**
  - [x] Update `OpticalFlowInterpolator` with `method="farneback" | "dis"`
  - [x] Implement `cv2.DISOpticalFlow` pipeline with ultrafast/medium/fine presets
  - [x] Benchmark and unit test DIS vs Farneback interpolation (passed)

- [x] **Task 4: Multi-Keyframe Procedural Animation (N >= 2 Keyframes)**
  - [x] Extend `ProceduralGenerator` to accept `keyframes: Sequence[np.ndarray]`
  - [x] Piecewise multi-segment SDF coordinate interpolation across keyframes
  - [x] Maintain 100% backwards compatibility with `(frame_a, frame_b)` signature
  - [x] Unit tests for 3+ keyframe smooth transition loops (passed)

- [x] **Task 5: Composable Processing Pipeline Engine**
  - [x] Create `engine/pipeline.py` with `Pipeline` and `StepTelemetry` abstractions
  - [x] Chain transforms: Ingest -> Optical Flow -> Smooth -> Crop -> Resize -> Motion Mask -> Mesh Export
  - [x] Step-by-step telemetry (execution time, array shape, metadata profile, summary report)
  - [x] Export in `engine/__init__.py` and comprehensive test coverage (passed)

---
