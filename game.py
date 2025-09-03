"""
Defines the core game objects for the simulation.

This module contains the classes for all entities that exist within the game
world, including the AI-controlled `Unit`, the `Target` for navigation, the
`Enemy` for combat, and the `Projectile` fired by units. It also includes a
`Wall` class, although walls are primarily handled by the `TileMap` class.
"""
import numpy as np
import pygame
from mlp import MLP
from quadtree import Rectangle

from map import Tile

class Unit:
    """Represents a single grid-based unit in the game, controlled by an MLP brain."""
    def __init__(self, id, grid_x, grid_y, tile_map, brain=None, whisker_length=10, perceivable_types=None):
        self.id = id
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.tile_map = tile_map

        self.color = (0, 150, 255)
        self.type = "unit"
        self.brain = brain
        self.whisker_length = whisker_length
        self.perceivable_types = perceivable_types if perceivable_types is not None else ["wall", "enemy", "target", "unit"]
        self.whisker_debug_info = []

        # 8 fixed whiskers for N, NE, E, SE, S, SW, W, NW
        self.whisker_directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        # In the new model, size is always 1 tile, used for rasterization
        self.size = 1

    def to_dict(self):
        """Serializes the unit's state to a dictionary for multiprocessing."""
        return {
            "id": self.id,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "type": self.type,
            "brain": self.brain,
            "num_whiskers": 8, # Fixed
            "whisker_length": self.whisker_length,
            "perceivable_types": self.perceivable_types,
        }

    def get_bounding_box(self):
        """Returns a 1x1 tile bounding box for the new grid-based model."""
        # The quadtree expects pixel coordinates.
        pixel_x = self.grid_x * self.tile_map.tile_size + self.tile_map.tile_size / 2
        pixel_y = self.grid_y * self.tile_map.tile_size + self.tile_map.tile_size / 2
        return Rectangle(pixel_x, pixel_y, self.tile_map.tile_size, self.tile_map.tile_size)

    def update(self, actions):
        """Updates the unit's grid position based on the MLP's chosen action."""
        # The action is the index of the whisker direction to move in.
        move_index = np.argmax(actions)

        dx, dy = self.whisker_directions[move_index]

        new_x = self.grid_x + dx
        new_y = self.grid_y + dy

        # Check for wall collisions before moving
        if self.tile_map.get_tile(new_x, new_y) != Tile.WALL:
            self.grid_x = new_x
            self.grid_y = new_y

    def draw(self, screen):
        """Draws the unit and its whiskers on the screen."""
        tile_size = self.tile_map.tile_size

        # Draw whiskers
        WHISKER_COLOR = (100, 100, 100)
        HIT_COLORS = {
            "wall": (200, 200, 200),
            "enemy": (255, 100, 100),
            "unit": (100, 100, 255),
            "target": (100, 255, 100),
        }
        for info in self.whisker_debug_info:
            start_pos = (info['start'][0] * tile_size + tile_size / 2, info['start'][1] * tile_size + tile_size / 2)
            end_pos = (info['end'][0] * tile_size + tile_size / 2, info['end'][1] * tile_size + tile_size / 2)
            full_end_pos = (info['full_end'][0] * tile_size + tile_size / 2, info['full_end'][1] * tile_size + tile_size / 2)

            pygame.draw.line(screen, WHISKER_COLOR, start_pos, full_end_pos, 1)
            if info['type']:
                hit_color = HIT_COLORS.get(info['type'], (255, 255, 255))
                pygame.draw.line(screen, hit_color, start_pos, end_pos, 2)
                pygame.draw.circle(screen, hit_color, end_pos, 4)

        # Draw unit body
        rect = pygame.Rect(self.grid_x * tile_size, self.grid_y * tile_size, tile_size, tile_size)
        pygame.draw.rect(screen, self.color, rect)

class Wall:
    """Represents a simple rectangular wall.

    Note:
        This class is largely unused, as walls are handled by the `TileMap`
        system for performance reasons. It is kept for potential future use
        or for representing dynamic wall objects.

    Attributes:
        rect (pygame.Rect): The rectangular area of the wall.
        color (tuple[int, int, int]): The display color of the wall.
        type (str): The object type identifier, "wall".
    """
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (100, 100, 100)
        self.type = "wall"

    def draw(self, screen):
        """Draws the wall on the screen.

        Args:
            screen (pygame.Surface): The pygame surface to draw on.
        """
        pygame.draw.rect(screen, self.color, self.rect)

class Target:
    """Represents the grid-based target point for the navigation training mode."""
    def __init__(self, grid_x, grid_y, tile_map):
        self.id = -1 # Static ID for the target
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.tile_map = tile_map
        self.color = (0, 255, 0)
        self.type = "target"
        # Size is now used for rasterization, representing tile radius
        self.size = 1

    def get_bounding_box(self):
        """Returns a 1x1 tile bounding box for the new grid-based model."""
        pixel_x = self.grid_x * self.tile_map.tile_size + self.tile_map.tile_size / 2
        pixel_y = self.grid_y * self.tile_map.tile_size + self.tile_map.tile_size / 2
        return Rectangle(pixel_x, pixel_y, self.tile_map.tile_size, self.tile_map.tile_size)

    def draw(self, screen):
        """Draws the target on the screen as a colored tile."""
        tile_size = self.tile_map.tile_size
        rect = pygame.Rect(self.grid_x * tile_size, self.grid_y * tile_size, tile_size, tile_size)
        pygame.draw.rect(screen, self.color, rect)

class Enemy:
    """Represents a stationary, grid-based enemy."""
    def __init__(self, grid_x, grid_y, tile_map):
        self.id = -2 # Static ID for the enemy
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.tile_map = tile_map
        self.color = (255, 0, 0)
        self.type = "enemy"
        # Size is now used for rasterization, representing tile radius
        self.size = 1

    def get_bounding_box(self):
        """Returns a 1x1 tile bounding box for the new grid-based model."""
        pixel_x = self.grid_x * self.tile_map.tile_size + self.tile_map.tile_size / 2
        pixel_y = self.grid_y * self.tile_map.tile_size + self.tile_map.tile_size / 2
        return Rectangle(pixel_x, pixel_y, self.tile_map.tile_size, self.tile_map.tile_size)

    def draw(self, screen):
        """Draws the enemy on the screen as a colored tile."""
        tile_size = self.tile_map.tile_size
        rect = pygame.Rect(self.grid_x * tile_size, self.grid_y * tile_size, tile_size, tile_size)
        pygame.draw.rect(screen, self.color, rect)

# The Projectile class has been removed as it is obsolete in the new grid-based model.

# Helper functions for geometry have been moved to math_utils.py
