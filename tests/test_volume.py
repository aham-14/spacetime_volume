"""Tests for engine/volume.py enhancements."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from engine.volume import SpaceTimeVolume


class TestVolume(unittest.TestCase):

    def test_volume_crop(self):
        data = np.arange(10 * 20 * 30, dtype=np.float32).reshape((10, 20, 30)) / (10 * 20 * 30)
        vol = SpaceTimeVolume(data, {"source": "test"})

        cropped = vol.crop(t=(2, 6), y=(5, 15), x=(10, 25))
        self.assertEqual(cropped.shape, (4, 10, 15))
        self.assertEqual(cropped.meta["parent_shape"], (10, 20, 30))
        np.testing.assert_array_equal(cropped.data, data[2:6, 5:15, 10:25])

    def test_volume_resize(self):
        data = np.ones((8, 16, 16), dtype=np.float32) * 0.5
        vol = SpaceTimeVolume(data)

        resized = vol.resize(new_shape=(4, 8, 8))
        self.assertEqual(resized.shape, (4, 8, 8))
        self.assertTrue(np.all(resized.data >= 0.0) and np.all(resized.data <= 1.0))

    def test_volume_histogram(self):
        data = np.linspace(0.0, 1.0, 1000, dtype=np.float32).reshape((10, 10, 10))
        vol = SpaceTimeVolume(data)

        counts, bin_edges = vol.histogram(bins=10)
        self.assertEqual(len(counts), 10)
        self.assertEqual(len(bin_edges), 11)
        self.assertEqual(counts.sum(), 1000)

    def test_volume_motion_mask(self):
        data = np.zeros((5, 10, 10), dtype=np.float32)
        # create motion at t=2
        data[2, 3:7, 3:7] = 0.8
        vol = SpaceTimeVolume(data)

        mask = vol.motion_mask(threshold=0.1)
        self.assertEqual(mask.shape, vol.shape)
        self.assertGreater(mask.data[2, 5, 5], 0.5)
        self.assertEqual(mask.data[0, 5, 5], 0.0)

    def test_volume_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_vol.npz"
            data = np.random.uniform(0.0, 1.0, (6, 12, 12)).astype(np.float32)
            meta = {"mode": "test", "tags": ["a", "b"], "step": 42}
            vol = SpaceTimeVolume(data, meta)

            saved_path = vol.save(tmp_path)
            self.assertTrue(saved_path.exists())

            loaded_vol = SpaceTimeVolume.load(saved_path)
            self.assertEqual(loaded_vol.shape, vol.shape)
            self.assertEqual(loaded_vol.meta["mode"], "test")
            self.assertEqual(loaded_vol.meta["step"], 42)
            np.testing.assert_allclose(loaded_vol.data, vol.data, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
