"""Tests for SLA resin hollowing and drain hole punching."""
import unittest
import numpy as np

from engine.volume import SpaceTimeVolume
from engine.mesh_export import MeshExporter


class TestHollowing(unittest.TestCase):

    def setUp(self):
        # Create solid cube
        sz = 30
        data = np.zeros((sz, sz, sz), dtype=np.float32)
        data[5:25, 5:25, 5:25] = 1.0
        self.vol = SpaceTimeVolume(data, {"shape": "cube"})

    def test_volume_hollowing(self):
        hollow_vol = self.vol.hollow(wall_thickness_vox=2.5, auto_drain=False)
        self.assertEqual(hollow_vol.shape, self.vol.shape)
        self.assertTrue(hollow_vol.meta["is_hollow"])

        solid_vox = np.sum(self.vol.data >= 0.5)
        hollow_vox = np.sum(hollow_vol.data >= 0.5)

        # Hollow volume should have substantially fewer interior voxels
        self.assertLess(hollow_vox, solid_vox)
        # Center should be hollow (0.0)
        self.assertEqual(hollow_vol.data[15, 15, 15], 0.0)
        # Outer boundary wall should remain solid (1.0)
        self.assertEqual(hollow_vol.data[6, 6, 6], 1.0)

    def test_auto_drain_holes(self):
        hollow_vol = self.vol.hollow(wall_thickness_vox=2.5, auto_drain=True, drain_hole_radius=2.0)
        self.assertTrue(hollow_vol.meta["has_drain_holes"])
        # Base slice at z=5 should have a drain hole cleared at the centroid
        self.assertEqual(hollow_vol.data[5, 15, 15], 0.0)


if __name__ == "__main__":
    unittest.main()
