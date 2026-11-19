"""Tests for constant geometric boundary enclosures & framing (Rounded Rect, Circle, Box)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.sdf_shapes import sdf_rect, sdf_rounded_rect
from engine.volume import SpaceTimeVolume
from engine.mesh_export import MeshExporter


class TestConstantBoundary(unittest.TestCase):

    def setUp(self):
        # Create a 3D synthetic volume: 16 frames, 48x48 resolution
        # Contains a small 10x10 motion patch at center
        self.T, self.H, self.W = 16, 48, 48
        data = np.zeros((self.T, self.H, self.W), dtype=np.float32)
        data[:, 19:29, 19:29] = 0.85
        self.vol = SpaceTimeVolume(data, {"name": "test_motion"})

    def test_sdf_rounded_rect_exact_math(self):
        # Grid 64x64, center 31.5, half-width 20, half-height 20, radius 5
        H, W = 64, 64
        cy, cx = 31.5, 31.5
        hh, hw = 20.0, 20.0
        r = 5.0
        d_rounded = sdf_rounded_rect(H, W, cy, cx, hh, hw, r)
        
        # At center, distance should be -20.0 (or -19.5 with grid discretization)
        self.assertAlmostEqual(d_rounded[31, 31], -19.5, delta=0.5)

        # On the axis edge (cy, cx + hw), point is on straight edge, distance ≈ 0
        x_edge = int(cx + hw)
        self.assertAlmostEqual(d_rounded[31, x_edge], 0.0, delta=0.6)

        # When r=0, sdf_rounded_rect must match sdf_rect exactly
        d_sharp_rect = sdf_rect(H, W, cy, cx, hh, hw)
        d_zero_r = sdf_rounded_rect(H, W, cy, cx, hh, hw, r=0.0)
        np.testing.assert_allclose(d_zero_r, d_sharp_rect, atol=1e-5)

    def test_apply_constant_boundary_framed_rounded_rect(self):
        bezel_vox = 3.5
        framed_vol = self.vol.apply_constant_boundary(
            shape="rounded_rect",
            mode="framed",
            bezel_voxels=bezel_vox,
            corner_radius=6.0,
            padding_voxels=2.0,
        )

        self.assertEqual(framed_vol.shape, self.vol.shape)
        self.assertTrue(framed_vol.meta["has_constant_boundary"])
        self.assertEqual(framed_vol.meta["boundary_shape"], "rounded_rect")
        self.assertEqual(framed_vol.meta["boundary_mode"], "framed")

        # Corner (0, 0) is well outside the rounded rect -> must be ~0.0
        self.assertLess(framed_vol.data[0, 0, 0], 0.01)

        # Center (24, 24) should preserve original video data (0.85)
        self.assertAlmostEqual(framed_vol.data[5, 24, 24], 0.85, delta=0.05)

        # Bezel rim near outer edge should be solid material (>= 0.8)
        # Center is (23.5, 23.5), hh/hw = 21.5, outer edge is ~45. Bezel is at ~43
        self.assertGreater(framed_vol.data[5, 24, 43], 0.8)

    def test_apply_constant_boundary_circle(self):
        circ_vol = self.vol.apply_constant_boundary(
            shape="circle",
            mode="framed",
            bezel_voxels=3.0,
            padding_voxels=2.0,
        )

        self.assertEqual(circ_vol.meta["boundary_shape"], "circle")
        # Grid corners outside circle radius must be ~0.0
        self.assertLess(circ_vol.data[0, 0, 0], 0.01)
        self.assertLess(circ_vol.data[0, self.H - 1, self.W - 1], 0.01)

        # Center video motion preserved
        self.assertAlmostEqual(circ_vol.data[0, 24, 24], 0.85, delta=0.05)

    def test_apply_constant_boundary_monolith(self):
        mono_vol = self.vol.apply_constant_boundary(
            shape="box",
            mode="monolith",
            padding_voxels=2.0,
        )
        self.assertEqual(mono_vol.meta["boundary_mode"], "monolith")
        # Voxels outside padding box are trimmed to 0.0
        self.assertLess(mono_vol.data[0, 0, 0], 0.01)
        # Interior video motion retained
        self.assertAlmostEqual(mono_vol.data[0, 24, 24], 0.85, delta=0.05)

    def test_apply_constant_boundary_solid_relief_and_base_plate(self):
        relief_vol = self.vol.apply_constant_boundary(
            shape="rounded_rect",
            mode="solid_relief",
            base_plate_slices=3,
        )
        self.assertEqual(relief_vol.meta["boundary_mode"], "solid_relief")
        self.assertEqual(relief_vol.meta["base_plate_slices"], 3)

        # Bottom 3 slices (0, 1, 2) should have solid floor raft (>= 0.95 at interior)
        self.assertGreaterEqual(relief_vol.data[0, 24, 24], 0.95)
        self.assertGreaterEqual(relief_vol.data[2, 24, 24], 0.95)

        # Later slice has relief density (> 0 inside footprint)
        self.assertGreater(relief_vol.data[6, 24, 24], 0.5)

    def test_mesh_exporter_integration(self):
        # Full export pipeline test to STL with rounded_rect constant boundary
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_framed_model.stl"
            exporter = MeshExporter(
                level=0.5,
                constant_boundary="rounded_rect",
                boundary_mode="framed",
                boundary_bezel=3.0,
                voxel_size=0.5,
            )
            stats = exporter.export(self.vol, out_file)

            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 1000)
            self.assertGreater(stats.n_vertices, 100)
            self.assertGreater(stats.n_faces, 100)
            self.assertTrue(stats.is_watertight)
            self.assertTrue(stats.fits_standard_sla_vat)


if __name__ == "__main__":
    unittest.main()
