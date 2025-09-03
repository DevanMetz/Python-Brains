import unittest
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_utils import vectorized_line_circle_intersection
from mlp_cupy import CUPY_AVAILABLE # Reuse the check for CuPy availability

if CUPY_AVAILABLE:
    import cupy as cp

def to_numpy(arr):
    """Helper to convert a CuPy array to NumPy, or no-op if it's already NumPy."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return arr.get()
    return arr

class TestVectorizedIntersection(unittest.TestCase):

    def _run_test_with_backend(self, xp):
        """
        A helper method to run a suite of intersection tests on a given backend (NumPy or CuPy).
        """
        # Test Case 1: Simple intersection
        p1s = xp.array([[0, 0]])
        p2s = xp.array([[10, 0]])
        centers = xp.array([[5, 0]])
        radii = xp.array([1])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        self.assertAlmostEqual(to_numpy(dists)[0], 4.0)
        self.assertEqual(to_numpy(idxs)[0], 0)

        # Test Case 2: No intersection
        centers = xp.array([[5, 5]])
        radii = xp.array([1])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        self.assertEqual(to_numpy(dists)[0], np.inf)
        self.assertEqual(to_numpy(idxs)[0], -1)

        # Test Case 3: Tangent intersection
        centers = xp.array([[5, 1]])
        radii = xp.array([1])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        self.assertAlmostEqual(to_numpy(dists)[0], 5.0)
        self.assertEqual(to_numpy(idxs)[0], 0)

        # Test Case 4: Intersection is off the line segment
        centers = xp.array([[15, 0]])
        radii = xp.array([1])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        self.assertEqual(to_numpy(dists)[0], np.inf)
        self.assertEqual(to_numpy(idxs)[0], -1)

        # Test Case 5: Multiple lines, one circle
        p1s = xp.array([[0, 0], [0, 5]])
        p2s = xp.array([[10, 0], [10, 5]])
        centers = xp.array([[5, 0]])
        radii = xp.array([2])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        self.assertAlmostEqual(to_numpy(dists)[0], 3.0)
        self.assertEqual(to_numpy(idxs)[0], 0)
        self.assertEqual(to_numpy(dists)[1], np.inf)
        self.assertEqual(to_numpy(idxs)[1], -1)

        # Test Case 6: One line, multiple circles (return closest)
        p1s = xp.array([[0, 0]])
        p2s = xp.array([[20, 0]])
        centers = xp.array([[5, 0], [15, 0]])
        radii = xp.array([2, 3])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        # Closest intersection is with the first circle (index 0) at distance 3.0
        self.assertAlmostEqual(to_numpy(dists)[0], 3.0)
        self.assertEqual(to_numpy(idxs)[0], 0)

        # Test Case 7: No objects to check against
        p1s = xp.array([[0, 0]])
        p2s = xp.array([[10, 0]])
        centers = xp.array([])
        radii = xp.array([])
        dists, idxs = vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp)
        self.assertEqual(to_numpy(dists)[0], np.inf)
        self.assertEqual(to_numpy(idxs)[0], -1)

    def test_numpy_backend(self):
        """Run all tests using the NumPy backend."""
        self._run_test_with_backend(np)

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy not available, skipping GPU tests")
    def test_cupy_backend(self):
        """Run all tests using the CuPy backend."""
        self._run_test_with_backend(cp)


if __name__ == '__main__':
    unittest.main()
