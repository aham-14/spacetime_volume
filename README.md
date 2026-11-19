# Space-Time Volume (4D to 3D Print & Casting Engine)

A modular, object-oriented Python backend and interactive 3D Jupyter UI that stacks 2D frames along a temporal Z-axis into continuous 3D scalar fields for high-resolution SLA resin printing and physical wax/silicone casting.

## Key Features

1. **Mode 1: Video Ingestion & Dense Optical Flow**
   - Ingests `.mp4` video files or raw numpy arrays into `(t, y, x)` float32 volumes.
   - **Farneback & DIS Optical Flow**: Synthesizes intermediate sub-frames via forward/backward displacement mapping to eliminate staggered voxel geometry and Z-axis aliasing. Includes **DIS (Dense Inverse Search)** flow for 10–50× faster sub-frame synthesis.
   - Light anisotropic Gaussian polish along the time axis.

2. **Mode 2: Procedural Generation (Multi-Keyframe SDF Morphing Engine)**
   - Generates 3D volumes procedurally from 2 or more keyframes (`N >= 2`).
   - **Continuous Signed Distance Fields (SDFs)**: Includes primitives (`disc`, `rect`, `ring`, `capsule`, `smooth_union`) and morphs directly in distance-field coordinate space before applying logistic sigmoids. This guarantees smooth sub-voxel gradients without staircasing.
   - Accepts 10 built-in mathematical easing curves (linear, quadratic, cubic, smoothstep, smootherstep, exponential, sine) or custom callables.

3. **Hardware Export Module (High-Res SLA Resin & Negative Mold Casting)**
   - **Multi-Format Export**: Native export to **STL (.stl)**, **Wavefront OBJ (.obj)**, **Polygon File (.ply)**, and **GL Transmission Format (.glb)**.
   - **Negative Mold Generation (Boolean Subtraction)**: Automatically builds a solid 3D bounding box and subtracts the Space-Time Volume in-place. Exports watertight, hollowed molds complete with top pour spouts for pouring wax or silicone.
   - **High-Resolution SLA Tuning**: Configurable `step_size` (default 1 for maximum resin resolution), dynamic `level` isosurface threshold selection, and reduced Taubin smoothing passes (default 4 iterations) to preserve fine pixel details.
   - **Strict Memory Management (8GB RAM / 4GB GPU)**: Chunked marching cubes processing with 1-voxel slice overlaps, SciPy sparse CSR slice occupancy indexing for $O(1)$ empty-chunk skipping, and Numba JIT parallelized in-place array operations.

4. **Composable Pipeline Engine (`engine/pipeline.py`)**
   - Method-chaining fluent API to construct reproducible generation and transformation workflows.
   - Built-in step profiling with execution timing, array dimensions, and metadata telemetry.

5. **3D Interactive Local Testing UI**
   - Built with `plotly.graph_objects.FigureWidget` and `ipywidgets`.
   - Projects frames dynamically onto a single flat `go.Surface` plane (no heavy voxel clouds).
   - Real-time time scrubbing, mode toggling, easing swapping, format selection, isosurface level slider, negative mold toggle, and background multi-threaded export.

## Architecture & Layout

```
spacetime_volume/
├── engine/                       # Pure backend (UI and HTTP agnostic)
│   ├── __init__.py               # Public API
│   ├── volume.py                 # SpaceTimeVolume (t,y,x) data container with crop/resize/save/load/mask
│   ├── video_ingest.py           # Mode 1: OpenCV video ingestor
│   ├── optical_flow.py           # Mode 1: Farneback & DIS sub-frame interpolation
│   ├── sdf_shapes.py             # SDF primitive library & sigmoid mapping
│   ├── procedural.py             # Mode 2: Multi-keyframe SDF morphing engine
│   ├── easing.py                 # 10 vectorised easing curves
│   ├── mesh_export.py            # Chunked marching cubes + Taubin + Negative Mold + Multi-format
│   ├── pipeline.py               # Composable processing pipeline engine
│   └── dummy_data.py             # Synthetic SDF keyframes & orbiting video stacks
├── tests/                        # Automated unit & integration tests
│   ├── test_volume.py
│   ├── test_mesh_export.py
│   ├── test_optical_flow.py
│   ├── test_procedural.py
│   └── test_pipeline.py
├── viewer.py                     # Plotly FigureWidget + ipywidgets control panel
├── app.py                        # Composition root
├── demo.ipynb                    # Interactive Jupyter notebook
└── requirements.txt
```

## Quick Start

### Installation

```bash
git clone https://github.com/aham-14/spacetime_volume.git
cd spacetime_volume
pip install -r requirements.txt
```

### Run Tests

Run the full automated test suite:
```bash
python -m unittest discover -s tests
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

### Composable Pipeline Example

```python
from engine import Pipeline, ProceduralGenerator, dummy_keyframes

fa, fb = dummy_keyframes()

pipe = (
    Pipeline()
    .then(lambda: ProceduralGenerator(fa, fb).generate(n_frames=24), name="generate")
    .smooth(sigma_t=1.0)
    .crop(t=(2, 22))
    .export_mesh("output/model.obj", level=0.5)
)

stats = pipe.run()
print(pipe.summary())
```
