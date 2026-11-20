import unittest

import numpy as np

from adaptive_placement import build_density_maps, sample_adaptive_locations


class AdaptivePlacementTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[0:96, 0:128]
        image = np.zeros((96, 128, 3), dtype=np.float32)
        # Add a textured/edge-rich region to make a non-constant feature map.
        image[20:76, 65:120, 0] = ((xx[20:76, 65:120] % 7) / 7.0)
        image[20:76, 65:120, 1] = ((yy[20:76, 65:120] % 9) / 9.0)
        image[20:76, 65:120, 2] = 0.8
        self.image = image

    def test_density_maps_are_finite_and_sized(self):
        rgb, edges, texture, density = build_density_maps(self.image)
        self.assertEqual(rgb.shape, (96, 128, 3))
        for feature in (edges, texture, density):
            self.assertEqual(feature.shape, (96, 128))
            self.assertTrue(np.isfinite(feature).all())
            self.assertGreaterEqual(float(feature.min()), 0.0)
            self.assertLessEqual(float(feature.max()), 1.0)

    def test_effective_count_matches_baseline_grid_count(self):
        locations = sample_adaptive_locations(
            self.image, n_strokes=200, seed=7
        )
        self.assertEqual(locations.shape, (196, 2))

    def test_seed_reproduces_positions(self):
        first = sample_adaptive_locations(self.image, 200, seed=123)
        second = sample_adaptive_locations(self.image, 200, seed=123)
        np.testing.assert_allclose(first, second)

    def test_locations_use_renderer_normalized_range(self):
        locations = sample_adaptive_locations(self.image, 200, seed=5)
        self.assertTrue(np.all(locations >= -0.95))
        self.assertTrue(np.all(locations <= 0.95))

    def test_constant_image_still_samples_due_to_epsilon_and_uniform_mix(self):
        constant = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        locations = sample_adaptive_locations(constant, 25, seed=1)
        self.assertEqual(locations.shape, (25, 2))
        self.assertTrue(np.isfinite(locations).all())


if __name__ == "__main__":
    unittest.main()
