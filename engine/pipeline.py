"""
engine/pipeline.py — Composable processing pipeline for Space-Time Volume workflows.

Allows chaining ingestion, optical-flow interpolation, temporal/spatial smoothing,
cropping, motion masking, and mesh export into reproducible, measurable steps.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .volume import SpaceTimeVolume


@dataclass
class StepTelemetry:
    """Diagnostic profile of a pipeline execution step."""
    name: str
    duration_s: float
    output_shape: Optional[Tuple[int, int, int]] = None
    info: dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Linear execution pipeline for SpaceTimeVolume transformations.

    Supports method chaining (.then(...)) or stage list initialization.

    Examples
    --------
    >>> pipe = (
    ...     Pipeline()
    ...     .then(lambda: ProceduralGenerator(fa, fb).generate(16), name="generate")
    ...     .smooth(sigma_t=1.0)
    ...     .crop(t=(2, 14))
    ... )
    >>> vol = pipe.run()
    """

    def __init__(
        self,
        *stages: Callable[..., Any],
        initial_input: Optional[Any] = None,
    ) -> None:
        self._stages: List[Tuple[str, Callable[..., Any]]] = []
        self._telemetry: List[StepTelemetry] = []
        self._initial_input = initial_input

        for s in stages:
            self.then(s)

    @property
    def telemetry(self) -> List[StepTelemetry]:
        """Telemetry records from the most recent run."""
        return list(self._telemetry)

    def summary(self) -> str:
        """Formatted execution report of pipeline stages."""
        if not self._telemetry:
            return "Pipeline has not run yet."
        lines = ["=== Pipeline Execution Summary ==="]
        total_time = 0.0
        for i, t in enumerate(self._telemetry, 1):
            total_time += t.duration_s
            shape_str = f"shape {t.output_shape}" if t.output_shape else "output non-volume"
            lines.append(f"  {i}. {t.name:<24} | {t.duration_s * 1000:6.1f} ms | {shape_str}")
        lines.append(f"Total time: {total_time:.3f} s")
        return "\n".join(lines)

    def then(
        self,
        stage: Callable[..., Any],
        name: Optional[str] = None,
    ) -> Pipeline:
        """Append a stage callable to the pipeline."""
        stage_name = name or getattr(stage, "__name__", repr(stage))
        self._stages.append((stage_name, stage))
        return self

    # --- Convenience Chainable Transformers ---

    def smooth(
        self,
        sigma_t: float = 1.5,
        sigma_y: float = 0.0,
        sigma_x: float = 0.0,
        truncate: float = 4.0,
    ) -> Pipeline:
        """Chain anisotropic Gaussian smoothing."""
        return self.then(
            lambda v: v.smoothed(sigma_t=sigma_t, sigma_y=sigma_y, sigma_x=sigma_x, truncate=truncate),
            name=f"smooth(σ_t={sigma_t})",
        )

    def crop(
        self,
        t: Optional[Tuple[int, int]] = None,
        y: Optional[Tuple[int, int]] = None,
        x: Optional[Tuple[int, int]] = None,
    ) -> Pipeline:
        """Chain 3D volume cropping."""
        return self.then(
            lambda v: v.crop(t=t, y=y, x=x),
            name="crop",
        )

    def resize(
        self,
        new_shape: Optional[Tuple[int, int, int]] = None,
        *,
        scale: Optional[Union[float, Tuple[float, float, float]]] = None,
    ) -> Pipeline:
        """Chain volume resampling."""
        return self.then(
            lambda v: v.resize(new_shape=new_shape, scale=scale),
            name="resize",
        )

    def motion_mask(self, threshold: float = 0.05) -> Pipeline:
        """Chain temporal motion difference masking."""
        return self.then(
            lambda v: v.motion_mask(threshold=threshold),
            name=f"motion_mask(th={threshold})",
        )

    def optical_flow(
        self,
        interp_factor: int = 4,
        method: str = "farneback",
    ) -> Pipeline:
        """Chain optical-flow sub-frame interpolation."""
        from .optical_flow import OpticalFlowInterpolator
        interpolator = OpticalFlowInterpolator(interp_factor=interp_factor, method=method)
        return self.then(
            interpolator.interpolate,
            name=f"optical_flow({method}, {interp_factor}x)",
        )

    def export_mesh(
        self,
        output_path: Union[str, Path],
        level: float = 0.5,
        step_size: int = 1,
        smooth_iterations: int = 4,
        as_negative_mold: bool = False,
    ) -> Pipeline:
        """Chain mesh or negative mold generation and export."""
        from .mesh_export import MeshExporter
        exporter = MeshExporter(
            level=level,
            step_size=step_size,
            smooth_iterations=smooth_iterations,
            as_negative_mold=as_negative_mold,
        )
        return self.then(
            lambda v: exporter.export(v, output_path=output_path),
            name=f"export_mesh({Path(output_path).name})",
        )

    # --- Execution ---

    def run(self, initial_input: Optional[Any] = None) -> Any:
        """Execute all stages sequentially, measuring execution profile."""
        current = initial_input if initial_input is not None else self._initial_input
        self._telemetry.clear()

        for name, stage in self._stages:
            t0 = time.perf_counter()
            if current is None:
                # Stage produces initial input (generator / ingestor)
                current = stage()
            else:
                current = stage(current)
            elapsed = time.perf_counter() - t0

            out_shape = current.shape if isinstance(current, SpaceTimeVolume) else None
            meta_info = current.meta if isinstance(current, SpaceTimeVolume) else {}

            self._telemetry.append(
                StepTelemetry(
                    name=name,
                    duration_s=elapsed,
                    output_shape=out_shape,
                    info=meta_info,
                )
            )

        return current
