import unittest
import sys
import os
import pygame

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_utils import iterative_line_circle_intersection

class TestIterativeIntersection(unittest.TestCase):

    def test_intersection_when_p1_is_inside_circle(self):
        """
        Tests for an intersection when the line segment starts inside the circle.
        This is the case where the bug occurs.
        """
        p1 = (1, 0)
        p2 = (10, 0)
        center = (0, 0)
        radius = 5

        # The intersection should be at (5, 0), which is a distance of 4 from p1.
        expected_distance = 4.0

        distance = iterative_line_circle_intersection(p1, p2, center, radius)

        self.assertIsNotNone(distance, "Intersection should be found, but was not.")
        self.assertAlmostEqual(distance, expected_distance, places=5)

if __name__ == '__main__':
    unittest.main()
