"""Tests for engine/procedural.py multi-keyframe SDF generation."""
import unittest
import numpy as np

from engine.procedural import ProceduralGenerator
from engine.sdf_shapes import sdf_disc, sdf_to_intensity


class TestProcedural(unittest.TestCase):

    def setUp(self):
        # Generate 3 keyframe discs moving across the frame: left -> center -> right
        h, w = 48, 64
        self.k0 = sdf_to_intensity(sdf_disc(h, w, cy=24, cx=16, r=10), edge_width=2.0)
        self.k1 = sdf_to_intensity(sdf_disc(h, w, cy=24, cx=32, r=14), edge_width=2.0)
        self.k2 = sdf_to_intensity(sdf_disc(h, w, cy=24, cx=48, r=8), edge_width=2.0)

    def test_two_keyframe_backwards_compatibility(self):
        gen = ProceduralGenerator(self.k0, self.k1)
        vol = gen.generate(n_frames=16, transition_function="linear (t)")

        self.assertEqual(vol.shape, (16, 48, 64))
        self.assertEqual(vol.meta["mode"], "procedural")
        self.assertEqual(vol.meta["n_keyframes"], 2)
        # Check start and end resemblance
        self.assertGreater(vol.data[0, 24, 16], 0.8)
        self.assertGreater(vol.data[-1, 24, 32], 0.8)

    def test_multi_keyframe_generation(self):
        gen = ProceduralGenerator(keyframes=[self.k0, self.k1, self.k2])
        vol = gen.generate(n_frames=30, transition_function="ease-in-out cubic")

        self.assertEqual(vol.shape, (30, 48, 64))
        self.assertEqual(vol.meta["n_keyframes"], 3)
        # First keyframe at t=0
        self.assertGreater(vol.data[0, 24, 16], 0.8)
        # Middle keyframe near t=15
        self.assertGreater(vol.data[15, 24, 32], 0.8)
        # Final keyframe at t=29
        self.assertGreater(vol.data[-1, 24, 48], 0.8)


if __name__ == "__main__":
    unittest.main()
