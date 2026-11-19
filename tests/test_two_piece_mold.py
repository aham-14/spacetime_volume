"""Tests for 2-piece mold generation with interlocking registration keys."""
import unittest
import numpy as np

from engine.volume import SpaceTimeVolume
from engine.mesh_export import generate_two_piece_mold, MeshExporter


class TestTwoPieceMold(unittest.TestCase):

    def setUp(self):
        sz = 20
        data = np.zeros((sz, sz, sz), dtype=np.float32)
        data[4:16, 4:16, 4:16] = 1.0
        self.vol = SpaceTimeVolume(data, {"shape": "box"})

    def test_two_piece_mold_split(self):
        part_a, part_b = generate_two_piece_mold(
            self.vol,
            wall_thickness=3,
            open_top=True,
            split_axis=1,
            key_radius=2,
            key_depth=2,
        )

        self.assertEqual(part_a.shape, part_b.shape)
        self.assertEqual(part_a.meta["mold_part"], "A")
        self.assertEqual(part_b.meta["mold_part"], "B")
        self.assertTrue(part_a.meta["has_registration_keys"])

        # Both parts should contain non-empty voxel data
        self.assertGreater(np.sum(part_a.data >= 0.5), 0)
        self.assertGreater(np.sum(part_b.data >= 0.5), 0)

        # Split plane check: Part A should be predominantly on one side, Part B on the other
        split_idx = part_a.meta["split_index"]
        # Far side of part A should be 0.0 (except registration keys)
        self.assertEqual(part_a.data[:, split_idx + 4:, :].sum(), 0.0)
        # Near side of part B should be 0.0
        self.assertEqual(part_b.data[:, :split_idx, :].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
