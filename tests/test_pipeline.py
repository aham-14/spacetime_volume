"""Tests for engine/pipeline.py composable pipeline engine."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from engine.volume import SpaceTimeVolume
from engine.pipeline import Pipeline, StepTelemetry
from engine.procedural import ProceduralGenerator
from engine.sdf_shapes import sdf_disc, sdf_to_intensity


class TestPipeline(unittest.TestCase):

    def test_pipeline_chaining_and_telemetry(self):
        h, w = 32, 32
        f0 = sdf_to_intensity(sdf_disc(h, w, cy=16, cx=10, r=6))
        f1 = sdf_to_intensity(sdf_disc(h, w, cy=16, cx=22, r=6))

        pipe = (
            Pipeline()
            .then(lambda: ProceduralGenerator(f0, f1).generate(n_frames=12), name="generate")
            .smooth(sigma_t=1.0)
            .crop(t=(2, 10))
            .resize(new_shape=(6, 16, 16))
        )

        vol = pipe.run()
        self.assertIsInstance(vol, SpaceTimeVolume)
        self.assertEqual(vol.shape, (6, 16, 16))

        telemetry = pipe.telemetry
        self.assertEqual(len(telemetry), 4)
        self.assertEqual(telemetry[0].name, "generate")
        self.assertEqual(telemetry[0].output_shape, (12, 32, 32))
        self.assertEqual(telemetry[-1].output_shape, (6, 16, 16))

        summary = pipe.summary()
        self.assertIn("Pipeline Execution Summary", summary)
        self.assertIn("Total time", summary)

    def test_pipeline_mesh_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_mesh = Path(tmp_dir) / "pipe_mesh.stl"
            data = np.zeros((16, 24, 24), dtype=np.float32)
            data[4:12, 6:18, 6:18] = 0.8
            vol = SpaceTimeVolume(data)

            pipe = (
                Pipeline(initial_input=vol)
                .smooth(sigma_t=0.5)
                .export_mesh(output_path=out_mesh, level=0.5, step_size=1, smooth_iterations=0)
            )

            stats = pipe.run()
            self.assertTrue(out_mesh.exists())
            self.assertGreater(stats.n_vertices, 0)


if __name__ == "__main__":
    unittest.main()
