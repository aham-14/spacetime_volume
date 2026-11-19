"""Tests for engine/mesh_export.py multi-format and level override enhancements."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from engine.volume import SpaceTimeVolume
from engine.mesh_export import MeshExporter, MeshStats


class TestMeshExport(unittest.TestCase):

    def setUp(self):
        # Create a sphere-like 3D volume
        sz = 24
        ys, xs, zs = np.mgrid[-1:1:complex(0, sz), -1:1:complex(0, sz), -1:1:complex(0, sz)]
        r = np.sqrt(xs**2 + ys**2 + zs**2)
        # Intensity: 1 inside radius 0.6, 0 outside, with smooth falloff
        data = np.clip(1.0 - (r - 0.4) / 0.4, 0.0, 1.0).astype(np.float32)
        self.vol = SpaceTimeVolume(data, {"name": "test_sphere"})

    def test_multi_format_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            exporter = MeshExporter(step_size=1, smooth_iterations=0)

            for ext in [".stl", ".obj", ".ply"]:
                out_path = tmp / f"sphere{ext}"
                stats = exporter.export(self.vol, out_path, level=0.5)

                self.assertTrue(out_path.exists())
                self.assertGreater(stats.n_vertices, 0)
                self.assertGreater(stats.n_faces, 0)
                self.assertEqual(stats.export_format, ext.lstrip("."))

    def test_level_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            exporter = MeshExporter(level=0.5, step_size=1, smooth_iterations=0)

            # Export with level=0.3 vs level=0.7
            stats_loose = exporter.export(self.vol, tmp / "sphere_03.stl", level=0.3)
            stats_tight = exporter.export(self.vol, tmp / "sphere_07.stl", level=0.7)

            self.assertTrue((tmp / "sphere_03.stl").exists())
            self.assertTrue((tmp / "sphere_07.stl").exists())
            # Lower threshold creates a larger volume
            self.assertGreater(stats_loose.volume_mm3, stats_tight.volume_mm3)


if __name__ == "__main__":
    unittest.main()
