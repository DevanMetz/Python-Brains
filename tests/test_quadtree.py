import unittest
import sys
import os

# Add the root directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quadtree import QuadTree, Rectangle

# A mock object class for testing purposes. It needs a way to be identified
# and a method to return its bounding box, just like our game objects.
class MockObject:
    def __init__(self, x, y, w=1, h=1, id_val=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = id_val

    def get_bounding_box(self):
        return Rectangle(self.x, self.y, self.w, self.h)

    # Make mock objects hashable so they can be put in a set
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, MockObject) and self.id == other.id

    def __repr__(self):
        return f"MockObject(id={self.id})"


class TestQuadTree(unittest.TestCase):

    def setUp(self):
        """Set up a fresh QuadTree for each test."""
        # A 200x200 world centered at (100,100), so it spans from (0,0) to (200,200)
        self.boundary = Rectangle(100, 100, 200, 200)
        self.qt = QuadTree(self.boundary, capacity=4)

    def test_initialization(self):
        """Test if the QuadTree is created with the correct boundary and capacity."""
        self.assertEqual(self.qt.boundary, self.boundary)
        self.assertEqual(self.qt.capacity, 4)
        self.assertFalse(self.qt.divided)
        self.assertEqual(len(self.qt.objects), 0)

    def test_insert_object(self):
        """Test that an object can be inserted correctly."""
        obj1 = MockObject(50, 50, id_val=1)
        self.assertTrue(self.qt.insert(obj1))
        self.assertIn(obj1, self.qt.objects)

    def test_insert_outside_boundary(self):
        """Test that an object outside the boundary is not inserted."""
        obj_outside = MockObject(300, 300, id_val=1)
        self.assertFalse(self.qt.insert(obj_outside))
        self.assertEqual(len(self.qt.objects), 0)

    def test_subdivision(self):
        """Test that the QuadTree subdivides when its capacity is exceeded."""
        # Insert 4 objects, should not subdivide yet
        for i in range(4):
            self.qt.insert(MockObject(10 + i*5, 10, id_val=i))

        self.assertEqual(len(self.qt.objects), 4)
        self.assertFalse(self.qt.divided)

        # Insert a 5th object, which should trigger subdivision
        self.qt.insert(MockObject(30, 10, id_val=4))
        self.assertTrue(self.qt.divided)
        self.assertEqual(len(self.qt.objects), 0) # Parent node should be empty
        self.assertIsNotNone(self.qt.northeast)

    def test_query_simple(self):
        """Test a simple query in a non-subdivided tree."""
        obj1 = MockObject(50, 50, id_val=1)
        obj2 = MockObject(150, 150, id_val=2)
        self.qt.insert(obj1)
        self.qt.insert(obj2)

        # Query a range that only contains obj1
        query_range = Rectangle(40, 40, 20, 20) # A 20x20 box centered at (40,40)
        found_objects = self.qt.query(query_range)
        self.assertEqual(len(found_objects), 1)
        self.assertIn(obj1, found_objects)

    def test_query_after_subdivision(self):
        """Test that queries correctly find objects in child nodes."""
        # These objects will be placed in different quadrants
        obj_nw = MockObject(50, 50, id_val=1)
        obj_ne = MockObject(150, 50, id_val=2)
        obj_sw = MockObject(50, 150, id_val=3)
        obj_se = MockObject(150, 150, id_val=4)
        obj_another_nw = MockObject(51, 51, id_val=5) # This will trigger subdivision

        self.qt.insert(obj_nw)
        self.qt.insert(obj_ne)
        self.qt.insert(obj_sw)
        self.qt.insert(obj_se)
        self.qt.insert(obj_another_nw)

        self.assertTrue(self.qt.divided)

        # Query the northwest quadrant
        query_nw = Rectangle(50, 50, 100, 100)
        found_nw = self.qt.query(query_nw)
        self.assertIn(obj_nw, found_nw)
        self.assertIn(obj_another_nw, found_nw)
        self.assertEqual(len(found_nw), 2)

        # Query a region with no objects
        query_empty = Rectangle(1, 1, 2, 2)
        found_empty = self.qt.query(query_empty)
        self.assertEqual(len(found_empty), 0)

    def test_query_spanning_object(self):
        """Test querying for an object that spans multiple quadrants."""
        # This object's bounding box will span all four quadrants
        spanning_obj = MockObject(100, 100, w=40, h=40, id_val=99)

        # Add other objects to force subdivision
        for i in range(5):
            self.qt.insert(MockObject(10 + i, 10 + i, id_val=i))

        self.qt.insert(spanning_obj)
        self.assertTrue(self.qt.divided)

        # A small query range at the center should find the spanning object
        query_center = Rectangle(100, 100, 10, 10)
        found = self.qt.query(query_center)
        self.assertIn(spanning_obj, found)

    def test_clear(self):
        """Test that the clear method resets the Quadtree."""
        for i in range(5):
            self.qt.insert(MockObject(10 + i, 10 + i, id_val=i))

        self.assertTrue(self.qt.divided)
        self.qt.clear()
        self.assertFalse(self.qt.divided)
        self.assertEqual(len(self.qt.objects), 0)
        self.assertIsNone(getattr(self.qt, 'northeast', None))


if __name__ == '__main__':
    unittest.main()
