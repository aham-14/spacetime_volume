# Space-Time Volume (4D to 3D Print & Casting Engine)

A modular, object-oriented Python backend and interactive 3D Jupyter UI that stacks 2D frames along a temporal Z-axis into continuous 3D scalar fields for high-resolution SLA resin printing and physical wax/silicone casting.

## Key Features

1. **Mode 1: Video Ingestion & Optical Flow (Standard Mode)**
   - Ingests `.mp4` video files or raw numpy arrays into `(t, y, x)` float32 volumes.
   - **Farneback Dense Optical Flow**: Synthesizes intermediate sub-frames via forward/backward displacement mapping to eliminate staggered voxel geometry and Z-axis aliasing.
   - Light anisotropic Gaussian polish along the time axis.

2. **Mode 2: Procedural Generation (SDF Morphing Engine)**
   - Generates 3D volumes procedurally from 2D keyframes.
   - **Continuous Signed Distance Fields (SDFs)**: Includes primitives (`disc`, `rect`, `ring`, `capsule`, `smooth_union`) and morphs directly in distance-field coordinate space before applying logistic sigmoids. This guarantees smooth sub-voxel gradients without staircasing.
   - Accepts 10 built-in mathematical easing curves (linear, quadratic, cubic, smoothstep, smootherstep, exponential, sine) or custom callables.

3. **Hardware Export Module (High-Res SLA Resin & Negative Mold Casting)**
   - **Negative Mold Generation (Boolean Subtraction)**: Automatically builds a solid 3D bounding box and subtracts the Space-Time Volume in-place. Exports watertight, hollowed STL molds complete with top pour spouts for pouring wax or silicone.
   - **High-Resolution SLA Tuning**: Exposes configurable `step_size` (default 1 for maximum resin resolution) and reduced Taubin smoothing passes (default 4 iterations) to preserve fine pixel details.
   - **Strict Memory Management (8GB RAM / 4GB GPU)**: Chunked marching cubes processing with 1-voxel slice overlaps, SciPy sparse CSR slice occupancy indexing for $O(1)$ empty-chunk skipping, and Numba JIT parallelized in-place array operations.

4. **3D Interactive Local Testing UI**
   - Built with `plotly.graph_objects.FigureWidget` and `ipywidgets`.
   - Projects frames dynamically onto a single flat `go.Surface` plane (no heavy voxel clouds).
   - Real-time time scrubbing, mode toggling, easing swapping, negative mold toggle, and background multi-threaded STL export.

## Architecture & Layout

```
spacetime_volume/
├── engine/                       # Pure backend (UI and HTTP agnostic)
│   ├── __init__.py               # Public API
│   ├── volume.py                 # SpaceTimeVolume (t,y,x) data container
│   ├── video_ingest.py           # Mode 1: OpenCV video ingestor
│   ├── optical_flow.py           # Mode 1: Farneback sub-frame interpolation
│   ├── sdf_shapes.py             # SDF primitive library & sigmoid mapping
│   ├── procedural.py             # Mode 2: SDF morphing with motion compensation
│   ├── easing.py                 # 10 vectorised easing curves
│   ├── mesh_export.py            # Chunked marching cubes + Taubin + Negative Mold
│   └── dummy_data.py             # Synthetic SDF keyframes & orbiting video stacks
├── viewer.py                     # Plotly FigureWidget + ipywidgets control panel
├── app.py                        # Composition root
├── demo.ipynb                    # 6-section interactive Jupyter notebook
└── requirements.txt
```

## Quick Start

### Installation

```bash
git clone https://github.com/aham-14/spacetime_volume.git
cd spacetime_volume
pip install -r requirements.txt
```

### Run Interactive Viewer

Launch Jupyter:
```bash
jupyter lab demo.ipynb
```

In the notebook:
```python
from app import build_viewer

viewer = build_viewer(
    video_path=None,         # None for synthetic demo, or r"path/to/clip.mp4"
    stl_step_size=1,         # 1 = Max SLA resin resolution
    stl_smooth_iters=4,      # 4 Taubin passes (preserves fine pixel features)
    chunk_size=32,           # Memory bounded for 8GB RAM host
    stl_output_dir="output",
)
viewer.show()
```

### Headless Mesh & Mold Export

```python
from engine import ProceduralGenerator, dummy_keyframes, MeshExporter

# 1. Generate Space-Time Volume
fa, fb = dummy_keyframes()
vol = ProceduralGenerator(fa, fb, edge_width=2.5).generate(48, "smoothstep (3t²−2t³)")

# 2. Export Positive SLA Model
exp_pos = MeshExporter(step_size=1, smooth_iterations=4, voxel_size=0.5)
stats_pos = exp_pos.export(vol, "output/positive_model.stl")

# 3. Export Negative Silicone/Wax Mold
exp_mold = MeshExporter(
    step_size=1,
    smooth_iterations=4,
    voxel_size=0.5,
    as_negative_mold=True,
    mold_wall_thickness=4,
    mold_open_top=True,
)
stats_mold = exp_mold.export(vol, "output/negative_mold.stl")
```

## License

MIT
