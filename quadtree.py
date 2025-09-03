import pygame

class Rectangle:
    """A simple rectangle class for Axis-Aligned Bounding Boxes."""
    def __init__(self, x, y, w, h):
        self.x = x  # Center x
        self.y = y  # Center y
        self.w = w
        self.h = h
        self.left = x - w / 2
        self.right = x + w / 2
        self.top = y - h / 2
        self.bottom = y + h / 2

    def intersects(self, other):
        """Check if this rectangle intersects with another rectangle."""
        return not (self.right < other.left or
                    other.right < self.left or
                    self.bottom < other.top or
                    other.bottom < self.top)

MAX_DEPTH = 8

class QuadTree:
    """
    A Quadtree data structure for 2D spatial partitioning.
    Objects are stored in leaf nodes. When a node's capacity is exceeded,
    it subdivides, and all of its objects are pushed down to the appropriate
    child nodes. Includes a max depth to prevent infinite recursion.
    """
    def __init__(self, boundary, capacity=4, depth=0):
        if not isinstance(boundary, Rectangle):
            raise TypeError("boundary must be a Rectangle object")
        self.boundary = boundary
        self.capacity = capacity
        self.objects = []
        self.divided = False
        self.depth = depth

    def subdivide(self):
        """Create four child QuadTrees."""
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w / 2
        h = self.boundary.h / 2

        new_depth = self.depth + 1
        ne = Rectangle(x + w / 2, y - h / 2, w, h)
        self.northeast = QuadTree(ne, self.capacity, new_depth)
        nw = Rectangle(x - w / 2, y - h / 2, w, h)
        self.northwest = QuadTree(nw, self.capacity, new_depth)
        se = Rectangle(x + w / 2, y + h / 2, w, h)
        self.southeast = QuadTree(se, self.capacity, new_depth)
        sw = Rectangle(x - w / 2, y + h / 2, w, h)
        self.southwest = QuadTree(sw, self.capacity, new_depth)

        self.divided = True

        # Move all objects from this node down to the new children
        for obj in self.objects:
            self.northeast.insert(obj)
            self.northwest.insert(obj)
            self.southeast.insert(obj)
            self.southwest.insert(obj)

        self.objects = []

    def insert(self, obj):
        """
        Insert an object into the Quadtree.
        An object must have a `get_bounding_box()` method that returns a Rectangle.
        """
        if not self.boundary.intersects(obj.get_bounding_box()):
            return False

        if self.divided:
            self.northeast.insert(obj)
            self.northwest.insert(obj)
            self.southeast.insert(obj)
            self.southwest.insert(obj)
            return True

        self.objects.append(obj)

        # If capacity is exceeded and we're not at max depth, subdivide.
        if len(self.objects) > self.capacity and self.depth < MAX_DEPTH:
            self.subdivide()

        return True

    def query(self, range_rect, found=None):
        """
        Query for objects within a given rectangular range.
        Returns a list of unique objects.
        """
        if found is None:
            found = set()

        # Do not check children if the query range doesn't intersect this node's boundary
        if not self.boundary.intersects(range_rect):
            return []

        if self.divided:
            self.northwest.query(range_rect, found)
            self.northeast.query(range_rect, found)
            self.southwest.query(range_rect, found)
            self.southeast.query(range_rect, found)
        else:
            # If this is a leaf node, check its objects
            for obj in self.objects:
                if range_rect.intersects(obj.get_bounding_box()):
                    found.add(obj)

        return list(found)

    def clear(self):
        """Clear the Quadtree for the next frame."""
        self.objects = []
        self.divided = False
        # Remove references to children to allow for garbage collection
        if hasattr(self, 'northeast'):
            del self.northeast
            del self.northwest
            del self.southeast
            del self.southwest

    def draw(self, surface, color=(255, 255, 255)):
        """Draw the Quadtree boundaries for debugging."""
        rect_coords = (self.boundary.left, self.boundary.top, self.boundary.w, self.boundary.h)
        pygame.draw.rect(surface, color, rect_coords, 1)

        if self.divided:
            self.northeast.draw(surface, color)
            self.northwest.draw(surface, color)
            self.southeast.draw(surface, color)
            self.southwest.draw(surface, color)
