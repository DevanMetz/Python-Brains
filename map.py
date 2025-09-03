import pygame
import numpy as np
from enum import Enum

class Tile(Enum):
    EMPTY = 0
    WALL = 1

class TileMap:
    """
    Manages the tile-based map for the simulation.
    """
    def __init__(self, width, height, tile_size):
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
        """Sets the type of a tile at a given grid coordinate."""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.grid[grid_x, grid_y] = tile_type

    def get_tile(self, grid_x, grid_y):
        """Gets the type of a tile at a given grid coordinate."""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.grid[grid_x, grid_y]
        return None # Out of bounds

    def get_tile_at_pixel(self, px, py):
        """Gets the type of a tile at a given pixel coordinate."""
        grid_x = int(px // self.tile_size)
        grid_y = int(py // self.tile_size)
        return self.get_tile(grid_x, grid_y)

    def get_wall_rects(self):
        """Returns a list of pygame.Rect objects for all wall tiles."""
        rects = []
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.grid[x, y] == Tile.WALL:
                    rects.append(pygame.Rect(x * self.tile_size, y * self.tile_size, self.tile_size, self.tile_size))
        return rects

    def draw(self, screen):
        """Draws the map grid and the wall tiles."""
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
