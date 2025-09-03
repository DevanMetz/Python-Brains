import math

class SpatialGrid:
    """
    A grid-based spatial partitioning system to accelerate object lookups.

    The world is divided into a grid of cells. Each object is registered
    into the cells it overlaps with. Queries for objects can then be
    restricted to a specific set of cells, avoiding a brute-force check
    against every object in the world.
    """
    def __init__(self, world_width, world_height, cell_size):
        """
        Initializes the spatial grid.

        Args:
            world_width (int): The pixel width of the simulation world.
            world_height (int): The pixel height of the simulation world.
            cell_size (int): The size of each square cell in pixels.
        """
        self.cell_size = cell_size
        self.grid_width = math.ceil(world_width / cell_size)
        self.grid_height = math.ceil(world_height / cell_size)
        self.grid = [set() for _ in range(self.grid_width * self.grid_height)]

    def _get_cell_indices(self, position, radius):
        """

        Gets all cell indices that an object's bounding box overlaps with.
        """
        min_x = max(0, int((position[0] - radius) / self.cell_size))
        max_x = min(self.grid_width - 1, int((position[0] + radius) / self.cell_size))
        min_y = max(0, int((position[1] - radius) / self.cell_size))
        max_y = min(self.grid_height - 1, int((position[1] + radius) / self.cell_size))

        indices = set()
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                indices.add(y * self.grid_width + x)
        return indices

    def clear(self):
        """
        Clears all objects from the grid. Should be called each frame.
        """
        for cell in self.grid:
            cell.clear()

    def register(self, obj_data):
        """
        Registers an object into the grid based on its position and size.

        Args:
            obj_data (dict): A dictionary containing the object's 'position' and 'size'.
        """
        pos = obj_data['position']
        size = obj_data['size']
        indices = self._get_cell_indices(pos, size)
        for index in indices:
            self.grid[index].add(obj_data['id'])


    def query_point(self, position):
        """
        Queries the grid for object IDs at a single point.
        """
        x = int(position.x / self.cell_size)
        y = int(position.y / self.cell_size)

        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            index = y * self.grid_width + x
            return self.grid[index]
        return set()

    def query_line(self, start_point, end_point):
        """
        Queries the grid for all object IDs in cells that a line segment intersects.

        Uses a DDA-like algorithm to step along the line and collect cells.
        """
        p1 = (start_point[0], start_point[1])
        p2 = (end_point[0], end_point[1])

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return self.query_point(start_point)

        x_inc, y_inc = dx / steps, dy / steps
        x, y = p1[0], p1[1]

        queried_ids = set()
        visited_cells = set()

        for _ in range(int(steps) + 1):
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)
            cell_index = grid_y * self.grid_width + grid_x

            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                if cell_index not in visited_cells:
                    queried_ids.update(self.grid[cell_index])
                    visited_cells.add(cell_index)
            x += x_inc
            y += y_inc

        return queried_ids
