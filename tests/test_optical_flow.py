"""Tests for engine/optical_flow.py Farneback and DIS optical flow interpolation."""
import unittest
import numpy as np

from engine.volume import SpaceTimeVolume
from engine.optical_flow import OpticalFlowInterpolator


class TestOpticalFlow(unittest.TestCase):

    def setUp(self):
        # Create a 2-frame volume with a moving square
        h, w = 40, 40
        f0 = np.zeros((h, w), dtype=np.float32)
        f1 = np.zeros((h, w), dtype=np.float32)

        f0[10:25, 10:25] = 1.0  # square at (10, 10)
        f1[15:30, 15:30] = 1.0  # shifted to (15, 15)

        data = np.stack([f0, f1], axis=0)
        self.vol = SpaceTimeVolume(data, {"source": "flow_test"})

    def test_farneback_interpolation(self):
        interp = OpticalFlowInterpolator(interp_factor=3, method="farneback")
        out_vol = interp.interpolate(self.vol)

        # Expected: (2 - 1) * 3 + 1 = 4 frames
        self.assertEqual(out_vol.shape, (4, 40, 40))
        self.assertEqual(out_vol.meta["optical_flow"], "farneback")
        # Middle sub-frame should contain non-zero pixels
        self.assertGreater(out_vol.data[1].sum(), 0)

    def test_dis_interpolation(self):
        interp = OpticalFlowInterpolator(interp_factor=3, method="dis", dis_preset=1)
        out_vol = interp.interpolate(self.vol)

        self.assertEqual(out_vol.shape, (4, 40, 40))
        self.assertEqual(out_vol.meta["optical_flow"], "dis")
        self.assertGreater(out_vol.data[1].sum(), 0)

    def test_flow_magnitude(self):
        mag_farneback = OpticalFlowInterpolator.flow_magnitude(self.vol, method="farneback")
        mag_dis = OpticalFlowInterpolator.flow_magnitude(self.vol, method="dis")

        self.assertEqual(len(mag_farneback), 1)
        self.assertEqual(len(mag_dis), 1)
        self.assertGreater(mag_farneback[0], 0.0)
        self.assertGreater(mag_dis[0], 0.0)


if __name__ == "__main__":
    unittest.main()
