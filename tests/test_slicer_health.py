"""Tests for automated mesh healing, physical dimensions, and SLA vat bounds check."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from engine.volume import SpaceTimeVolume
from engine.mesh_export import MeshExporter


class TestSlicerHealth(unittest.TestCase):

    def setUp(self):
        sz = 24
        data = np.zeros((sz, sz, sz), dtype=np.float32)
        data[4:20, 6:18, 6:18] = 0.9
        self.vol = SpaceTimeVolume(data)

    def test_slicer_health_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_mesh.stl"
            exporter = MeshExporter(voxel_size=0.5, step_size=1, smooth_iterations=2)
            stats = exporter.export(self.vol, out_file, repair_mesh=True)

            self.assertTrue(out_file.exists())
            self.assertGreater(stats.dimensions_mm[0], 0.0)
            self.assertGreater(stats.dimensions_mm[1], 0.0)
            self.assertGreater(stats.dimensions_mm[2], 0.0)

            # Voxel size is 0.5mm, 24 voxels = max 12mm. Fits well within 143 x 89 x 175 mm
            self.assertTrue(stats.fits_standard_sla_vat)
            self.assertTrue(stats.repair_applied)


if __name__ == "__main__":
    unittest.main()
