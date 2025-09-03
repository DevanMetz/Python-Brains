"""
Defines and manages the tile-based map for the simulation world.

This module provides the `TileMap` class, which is responsible for creating,
updating, and drawing the grid of static tiles (walls and empty space) that
form the environment for the simulation.
"""
import pygame
import numpy as np
from enum import Enum

class Tile(Enum):
    """An enumeration for the different types of tiles."""
    EMPTY = 0
    WALL = 1

class TileMap:
    """Manages the tile-based map for the simulation.

    This class uses a NumPy array to efficiently store the grid of tiles. It
    provides methods for converting between pixel coordinates and grid
    coordinates, modifying tile states, and drawing the map to the screen.

    Attributes:
        pixel_width (int): The total width of the map in pixels.
        pixel_height (int): The total height of the map in pixels.
        tile_size (int): The side length of a single square tile in pixels.
        grid_width (int): The width of the map in number of tiles.
        grid_height (int): The height of the map in number of tiles.
        grid (np.ndarray): A 2D NumPy array holding the state (`Tile` enum)
            of each tile in the map.
    """
    def __init__(self, width, height, tile_size):
        """Initializes the TileMap.

        Args:
            width (int): The desired width of the map in pixels.
            height (int): The desired height of the map in pixels.
            tile_size (int): The size of each square tile in pixels.
        """
        self.pixel_width = width
        self.pixel_height = height
        self.tile_size = tile_size
        self.grid_width = width // tile_size
        self.grid_height = height // tile_size

        # Create a 2D numpy array to hold the grid data
        self.grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY, dtype=Tile)

        # Colors
        self.wall_color = (100, 100, 100)
        self.grid_color = (40, 40, 40)

    def set_tile(self, grid_x, grid_y, tile_type):
        """Sets the type of a tile at a given grid coordinate.

        Performs bounds checking to ensure the coordinates are valid.

        Args:
            grid_x (int): The x-coordinate on the grid (column).
            grid_y (int): The y-coordinate on the grid (row).
            tile_type (Tile): The type of tile to set (e.g., `Tile.WALL`).
        """
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.grid[grid_x, grid_y] = tile_type

    def get_tile(self, grid_x, grid_y):
        """Gets the type of a tile at a given grid coordinate.

        Performs bounds checking.

        Args:
            grid_x (int): The x-coordinate on the grid (column).
            grid_y (int): The y-coordinate on the grid (row).

        Returns:
            Tile or None: The `Tile` enum member at the given coordinate, or
            `None` if the coordinates are out of bounds.
        """
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.grid[grid_x, grid_y]
        return None # Out of bounds

    def get_tile_at_pixel(self, px, py):
        """Gets the type of a tile at a given pixel coordinate.

        Args:
            px (float): The x-coordinate in pixels.
            py (float): The y-coordinate in pixels.

        Returns:
            Tile or None: The `Tile` enum member at the location of the pixel,
            or `None` if the coordinates are out of bounds.
        """
        grid_x = int(px // self.tile_size)
        grid_y = int(py // self.tile_size)
        return self.get_tile(grid_x, grid_y)

    def get_wall_rects(self):
        """Returns a list of pygame.Rect objects for all wall tiles.

        This can be useful for optimized collision detection systems that
        operate on lists of rectangles.

        Returns:
            list[pygame.Rect]: A list of rectangles representing all walls.
        """
        rects = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.grid[x, y] == Tile.WALL:
                    rects.append(pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))
        return rects

    def draw(self, screen):
        """Draws the map grid and the wall tiles onto a surface.

        Args:
            screen (pygame.Surface): The pygame surface to draw on.
        """
        # Draw walls
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.grid[x, y] == Tile.WALL:
                    rect = pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size)
                    pygame.draw.rect(screen, self.wall_color, rect)

        # Draw grid lines
        for x in range(0, self.pixel_width, self.tile_size):
            pygame.draw.line(screen, self.grid_color, (x, 0), (x, self.pixel_height))
        for y in range(0, self.pixel_height, self.tile_size):
            pygame.draw.line(screen, self.grid_color, (0, y), (self.pixel_width, y))
