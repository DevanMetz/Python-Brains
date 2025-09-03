"""
Provides a Quadtree data structure for efficient 2D spatial partitioning.

This module is a key performance optimization for the simulation. It contains
the `QuadTree` class, which allows for rapid querying of objects within a
specific area, avoiding the need to check every object in the world for
interactions like collision detection or sensory perception. It also includes
a helper `Rectangle` class for defining bounding boxes.
"""
import pygame

class Rectangle:
    """A simple rectangle class for representing Axis-Aligned Bounding Boxes.

    This class defines a rectangle by its center point (x, y) and its full
    width and height. It pre-calculates the coordinates of its edges for
    efficient intersection testing.

    Attributes:
        x (float): The x-coordinate of the rectangle's center.
        y (float): The y-coordinate of the rectangle's center.
        w (float): The total width of the rectangle.
        h (float): The total height of the rectangle.
        left (float): The x-coordinate of the left edge.
        right (float): The x-coordinate of the right edge.
        top (float): The y-coordinate of the top edge.
        bottom (float): The y-coordinate of the bottom edge.
    """
    def __init__(self, x, y, w, h):
        """Initializes the Rectangle.

        Args:
            x (float): The x-coordinate of the center.
            y (float): The y-coordinate of the center.
            w (float): The width.
            h (float): The height.
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.left = x - w / 2
        self.right = x + w / 2
        self.top = y - h / 2
        self.bottom = y + h / 2

    def intersects(self, other):
        """Checks if this rectangle intersects with another rectangle.

        Args:
            other (Rectangle): The other rectangle to check for intersection.

        Returns:
            bool: True if the rectangles overlap, False otherwise.
        """
        return not (self.right < other.left or
                    other.right < self.left or
                    self.bottom < other.top or
                    other.bottom < self.top)

MAX_DEPTH = 5

class QuadTree:
    """A Quadtree data structure for 2D spatial partitioning.

    This class recursively subdivides a 2D space into four quadrants,
    organizing objects into a tree structure based on their location. This
    allows for efficient spatial queries, drastically reducing the number of
    object-to-object comparisons needed for tasks like collision detection.

    Objects are stored in leaf nodes. When a node's capacity is exceeded, it
    subdivides, and all of its objects are pushed down to the appropriate
    child nodes. A maximum depth is used to prevent infinite recursion in
    cases of high object density.

    Attributes:
        boundary (Rectangle): The rectangular area this node covers.
        capacity (int): The maximum number of objects a node can hold before
            it must subdivide.
        objects (list): The list of objects contained in this node. This is
            only populated for leaf nodes.
        divided (bool): A flag indicating whether this node has subdivided.
        depth (int): The depth of this node in the tree.
    """
    def __init__(self, boundary, capacity=4, depth=0):
        """Initializes the QuadTree node.

        Args:
            boundary (Rectangle): The rectangular boundary of this quadtree node.
            capacity (int, optional): The maximum number of objects this node
                can hold before subdividing. Defaults to 4.
            depth (int, optional): The current depth of this node in the tree.
                Used to limit recursion. Defaults to 0.

        Raises:
            TypeError: If the boundary is not a Rectangle object.
        """
        if not isinstance(boundary, Rectangle):
            raise TypeError("boundary must be a Rectangle object")
        self.boundary = boundary
        self.capacity = capacity
        self.objects = []
        self.divided = False
        self.depth = depth

    def subdivide(self):
        """Create four child QuadTrees, dividing this node into quadrants."""
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
        """Inserts an object into the Quadtree.

        If the object does not fit within this node's boundary, it is not
        inserted. If the node has capacity, the object is added to its list.
        If the node is full, it subdivides, and the object is inserted into
        the appropriate child node(s).

        Args:
            obj: The object to insert. It must have a `get_bounding_box()`
                 method that returns a `Rectangle`.

        Returns:
            bool: True if the object was inserted successfully, False otherwise.
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
        """Queries for all objects within a given rectangular range.

        This method recursively searches the tree, checking only the quadrants
        that overlap with the query range. This is far more efficient than a
        brute-force search of all objects.

        Args:
            range_rect (Rectangle): The rectangular area to query.
            found (set, optional): A set used internally during recursion to
                collect unique objects. Users should not set this.

        Returns:
            list: A list of unique objects found within the query range.
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
        """Clears the entire Quadtree for the next simulation frame.

        This method removes all objects and children from the tree, resetting
        it to a single, empty root node. This should be called at the
        beginning of each frame before re-inserting all objects.
        """
        self.objects = []
        self.divided = False
        # Remove references to children to allow for garbage collection
        if hasattr(self, 'northeast'):
            del self.northeast
            del self.northwest
            del self.southeast
            del self.southwest

    def draw(self, surface, color=(255, 255, 255)):
        """Draws the Quadtree's boundaries for debugging purposes.

        Args:
            surface (pygame.Surface): The pygame surface to draw on.
            color (tuple, optional): The color of the boundary lines.
                Defaults to (255, 255, 255).
        """
        rect_coords = (self.boundary.left, self.boundary.top, self.boundary.w, self.boundary.h)
        pygame.draw.rect(surface, color, rect_coords, 1)

        if self.divided:
            self.northeast.draw(surface, color)
            self.northwest.draw(surface, color)
            self.southeast.draw(surface, color)
            self.southwest.draw(surface, color)
