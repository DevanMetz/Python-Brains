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
    """Represents a single unit in the game, controlled by an MLP brain.

    This is the main actor in the simulation. A unit has a position, velocity,
    and a neural network "brain" that dictates its actions. It perceives the
    world through a series of "whiskers" (raycasts) and uses the information
    gathered to make decisions about movement and attacking.

    Attributes:
        id (int): A unique identifier for the unit.
        position (pygame.Vector2): The unit's current center coordinates.
        tile_map (TileMap): A reference to the main tile map for collision checks.
        angle (float): The unit's current facing direction in radians.
        velocity (pygame.Vector2): The unit's current velocity vector.
        size (int): The radius of the unit's circular body.
        color (tuple[int, int, int]): The display color of the unit.
        speed (float): The maximum speed of the unit.
        type (str): The object type identifier, always "unit".
        damage_dealt (int): A counter for the total damage dealt by this unit.
        brain (MLP): The neural network controlling the unit's actions.
        num_whiskers (int): The number of sensory whiskers.
        whisker_length (int): The maximum range of the whiskers.
        perceivable_types (list[str]): A list of object types the unit can "see".
        attack_cooldown (int): A timer to limit the rate of fire.
        max_cooldown (int): The value to which the attack cooldown is reset.
        whisker_angles (np.ndarray): An array of whisker angles relative to the
            unit's facing direction.
        whisker_debug_info (list[dict]): Data for visualizing whisker casts.
    """
    def __init__(self, id, x, y, tile_map, brain=None, num_whiskers=7, whisker_length=150, perceivable_types=None):
        """Initializes a Unit instance.

        Args:
            id (int): The unique ID for the unit.
            x (float): The initial x-coordinate.
            y (float): The initial y-coordinate.
            tile_map (TileMap): The simulation's tile map.
            brain (MLP, optional): A pre-existing brain. If None, a new one is
                created based on the unit's configuration. Defaults to None.
            num_whiskers (int, optional): The number of sensory whiskers.
                Defaults to 7.
            whisker_length (int, optional): The maximum length of whiskers.
                Defaults to 150.
            perceivable_types (list[str], optional): A list of object types
                the unit can perceive. Defaults to a standard set.
        """
        self.id = id
        self.position = pygame.Vector2(x, y)
        self.tile_map = tile_map
        self.angle = np.random.uniform(0, 2 * np.pi) # Facing direction in radians
        self.velocity = pygame.Vector2(0, 0)
        self.size = 10
        self.color = (0, 150, 255)
        self.speed = 2.0
        self.type = "unit"
        self.damage_dealt = 0

        self.brain = brain
        self.num_whiskers = num_whiskers
        self.whisker_length = whisker_length
        self.perceivable_types = perceivable_types if perceivable_types is not None else ["wall", "enemy", "target", "unit"]

        self.attack_cooldown = 0
        self.max_cooldown = 30

        if self.num_whiskers > 1:
            self.whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_whiskers)
        else:
            self.whisker_angles = np.array([0]) # Single whisker facing forward
        self.whisker_debug_info = [] # To store data for drawing

    def to_dict(self):
        """Serializes the unit's state to a dictionary for multiprocessing.

        This method converts the essential parts of the unit's state into a
        simple dictionary format. This is necessary so that the unit's data
        can be "pickled" and sent to worker processes in the multiprocessing
        pool. The `pygame.Vector2` objects are converted to tuples. The brain
        is passed directly as NumPy arrays are picklable.

        Returns:
            dict: A serializable dictionary representing the unit's state.
        """
        return {
            "id": self.id,
            "position": (self.position.x, self.position.y),
            "angle": self.angle,
            "velocity": (self.velocity.x, self.velocity.y),
            "size": self.size,
            "type": self.type,
            "brain": self.brain,
            "num_whiskers": self.num_whiskers,
            "whisker_length": self.whisker_length,
            "perceivable_types": self.perceivable_types,
        }

    def get_bounding_box(self):
        """Returns a Rectangle representing the unit's bounding box.

        This is used by the Quadtree for spatial partitioning and collision
        detection optimization.

        Returns:
            Rectangle: An axis-aligned bounding box for the unit.
        """
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def get_inputs(self, world_objects, target):
        """
        Calculates inputs for the MLP brain, including whisker data and target vector.

        Note:
            This method is a legacy implementation and is no longer called
            directly in the main simulation loop, which now uses the optimized,
            vectorized `get_unit_inputs` function in `trainer.py`. It is kept
            for potential debugging, single-threaded modes, or as a fallback.
        """
        self.whisker_debug_info = []
        num_perceivables = len(self.perceivable_types)
        whisker_inputs = np.zeros((self.num_whiskers, num_perceivables))

        for i, whisker_angle in enumerate(self.whisker_angles):
            abs_angle = self.angle + whisker_angle
            start_point = self.position
            end_point = self.position + pygame.Vector2(self.whisker_length, 0).rotate(np.rad2deg(abs_angle))

            closest_dist = self.whisker_length
            detected_type = None
            closest_intersect_point = end_point

            # --- Detect Walls from TileMap using DDA algorithm ---
            if "wall" in self.perceivable_types:
                dx = end_point.x - start_point.x
                dy = end_point.y - start_point.y

                steps = max(abs(dx), abs(dy))
                if steps > 0:
                    x_inc = dx / steps
                    y_inc = dy / steps

                    x, y = start_point.x, start_point.y
                    for _ in range(int(steps)):
                        if self.tile_map.get_tile_at_pixel(x, y) == Tile.WALL:
                            check_point = pygame.Vector2(x, y)
                            closest_dist = self.position.distance_to(check_point)
                            detected_type = "wall"
                            closest_intersect_point = check_point
                            break
                        x += x_inc
                        y += y_inc

            # --- Detect other objects (enemies, targets, etc.) ---
            for obj in world_objects:
                if obj is self:
                    continue
                # This is a placeholder for the actual line_circle_intersection function
                # which was moved to math_utils.py. The logic here is illustrative.
                dist = None # line_circle_intersection(start_point, end_point, obj.position, obj.size)
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    detected_type = obj.type
                    closest_intersect_point = start_point + pygame.Vector2(dist, 0).rotate(np.rad2deg(abs_angle))


            self.whisker_debug_info.append({
                'start': start_point, 'end': closest_intersect_point,
                'full_end': end_point, 'type': detected_type
            })

            if detected_type and detected_type in self.perceivable_types:
                type_index = self.perceivable_types.index(detected_type)
                whisker_inputs[i, type_index] = 1.0 - (closest_dist / self.whisker_length)

        # --- Calculate relative target vector ---
        relative_vec = target.position - self.position
        # Rotate vector to be relative to the unit's orientation
        relative_vec = relative_vec.rotate(-np.rad2deg(self.angle))
        # Normalize
        norm_dx = relative_vec.x / self.tile_map.pixel_width
        norm_dy = relative_vec.y / self.tile_map.pixel_height
        target_inputs = np.array([norm_dx, norm_dy])

        # --- Concatenate all inputs ---
        flat_whisker_inputs = whisker_inputs.flatten()
        other_inputs = np.array([self.velocity.length() / self.speed, self.angle / (2 * np.pi)])
        return np.concatenate((flat_whisker_inputs, other_inputs, target_inputs))

    def attack(self):
        """Creates and returns a new projectile if the attack cooldown is over.

        Returns:
            Projectile or None: A new Projectile object if one can be fired,
            otherwise None.
        """
        if self.attack_cooldown <= 0:
            self.attack_cooldown = self.max_cooldown
            return Projectile(self.position.x, self.position.y, self.angle, self)
        return None

    def _check_wall_collision(self, position):
        """Checks if a given position collides with a wall tile.

        This method checks the four corners of the unit's bounding box against
        the tile map to determine if the unit would be inside a wall at the
        given position.

        Args:
            position (pygame.Vector2): The position to check.

        Returns:
            bool: True if the position collides with a wall, False otherwise.
        """
        # Check the four corners of the unit's bounding box
        points_to_check = [
            (position.x - self.size, position.y - self.size),
            (position.x + self.size, position.y - self.size),
            (position.x - self.size, position.y + self.size),
            (position.x + self.size, position.y + self.size),
        ]
        for p in points_to_check:
            if self.tile_map.get_tile_at_pixel(p[0], p[1]) == Tile.WALL:
                return True
        return False

    def update(self, actions, world_projectiles):
        """Updates the unit's state based on actions from its MLP brain.

        This method translates the numerical output of the MLP into game
        actions. It handles movement, turning, wall collision with sliding,
        and firing projectiles.

        Args:
            actions (np.ndarray): The output array from the MLP's forward pass.
                Expected shape is (1, 2) or (1, 3).
            world_projectiles (list[Projectile]): The main list of active
                projectiles in the simulation, to which a new one may be added.
        """
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # --- Calculate movement from MLP actions ---
        turn_action = actions[0][0]
        move_action = actions[0][1]
        self.angle += turn_action * 0.1
        forward_speed = max(0, move_action) * self.speed
        self.velocity = pygame.Vector2(forward_speed, 0).rotate(np.rad2deg(self.angle))

        # --- Handle collision and sliding ---
        # Move on x-axis
        new_pos_x = self.position.copy()
        new_pos_x.x += self.velocity.x
        if not self._check_wall_collision(new_pos_x):
            self.position.x = new_pos_x.x

        # Move on y-axis
        new_pos_y = self.position.copy()
        new_pos_y.y += self.velocity.y
        if not self._check_wall_collision(new_pos_y):
            self.position.y = new_pos_y.y

        # --- Handle attack action ---
        if len(actions[0]) > 2:
            attack_action = actions[0][2]
            if attack_action > 0.5: # Threshold to fire
                projectile = self.attack()
                if projectile:
                    world_projectiles.append(projectile)

    def draw(self, screen):
        """Draws the unit and its debug visualizations on the screen.

        This includes the unit's body, a direction indicator, and its
        sensory whiskers, which are color-coded based on what they are
        detecting.

        Args:
            screen (pygame.Surface): The pygame surface to draw on.
        """
        # --- Draw whiskers first, so they are behind the unit ---
        WHISKER_COLOR = (100, 100, 100) # Faint gray for max range
        HIT_COLORS = {
            "wall": (200, 200, 200), # White
            "enemy": (255, 100, 100), # Light Red
            "unit": (100, 100, 255), # Light Blue
            "target": (100, 255, 100), # Light Green
        }

        for info in self.whisker_debug_info:
            # Draw the full length whisker faintly
            pygame.draw.line(screen, WHISKER_COLOR, info['start'], info['full_end'], 1)

            # If there was a hit, draw the collision line more brightly
            if info['type']:
                hit_color = HIT_COLORS.get(info['type'], (255, 255, 255)) # Default to bright white
                pygame.draw.line(screen, hit_color, info['start'], info['end'], 2)
                pygame.draw.circle(screen, hit_color, (int(info['end'].x), int(info['end'].y)), 3)


        # --- Draw unit body ---
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

        # --- Draw direction indicator ---
        end_pos = self.position + pygame.Vector2(self.size, 0).rotate(np.rad2deg(self.angle))
        pygame.draw.line(screen, (255, 0, 0), self.position, end_pos, 2)

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
    """Represents the target point for the navigation training mode.

    Attributes:
        id (int): A static identifier for the target.
        position (pygame.Vector2): The target's coordinates.
        size (int): The radius of the target's circular shape.
        color (tuple[int, int, int]): The display color of the target.
        type (str): The object type identifier, "target".
    """
    def __init__(self, x, y):
        """Initializes a Target instance.

        Args:
            x (float): The initial x-coordinate.
            y (float): The initial y-coordinate.
        """
        self.id = -1 # Static ID for the target
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (0, 255, 0)
        self.type = "target"

    def to_dict(self):
        """Serializes the target's state to a dictionary.

        Returns:
            dict: A serializable dictionary representing the target's state.
        """
        return {
            "id": self.id,
            "position": (self.position.x, self.position.y),
            "size": self.size,
            "type": self.type,
        }

    def get_bounding_box(self):
        """Returns a Rectangle representing the target's bounding box.

        Returns:
            Rectangle: An axis-aligned bounding box for the target.
        """
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def draw(self, screen):
        """Draws the target on the screen.

        Args:
            screen (pygame.Surface): The pygame surface to draw on.
        """
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

class Enemy:
    """Represents a stationary enemy for the combat training mode.

    Attributes:
        id (int): A static identifier for the enemy.
        position (pygame.Vector2): The enemy's coordinates.
        size (int): The radius of the enemy's circular body.
        color (tuple[int, int, int]): The display color of the enemy.
        health (int): The current health of the enemy.
        type (str): The object type identifier, "enemy".
    """
    def __init__(self, x, y):
        """Initializes an Enemy instance.

        Args:
            x (float): The initial x-coordinate.
            y (float): The initial y-coordinate.
        """
        self.id = -2 # Static ID for the enemy
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (255, 0, 0)
        self.health = 100
        self.type = "enemy"

    def to_dict(self):
        """Serializes the enemy's state to a dictionary.

        Returns:
            dict: A serializable dictionary representing the enemy's state.
        """
        return {
            "id": self.id,
            "position": (self.position.x, self.position.y),
            "size": self.size,
            "type": self.type,
            "health": self.health,
        }

    def get_bounding_box(self):
        """Returns a Rectangle representing the enemy's bounding box.

        Returns:
            Rectangle: An axis-aligned bounding box for the enemy.
        """
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def draw(self, screen):
        """Draws the enemy and its health bar on the screen.

        Args:
            screen (pygame.Surface): The pygame surface to draw on.
        """
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)
        # Draw health bar
        if self.health < 100:
            bar_width = self.size * 2
            bar_height = 5
            health_frac = self.health / 100.0
            pygame.draw.rect(screen, (255,0,0), (self.position.x - self.size, self.position.y - self.size - 10, bar_width, bar_height))
            pygame.draw.rect(screen, (0,255,0), (self.position.x - self.size, self.position.y - self.size - 10, bar_width * health_frac, bar_height))

class Projectile:
    """Represents a projectile fired by a unit.

    Projectiles are simple, straight-moving entities with a limited lifespan.

    Attributes:
        position (pygame.Vector2): The projectile's current coordinates.
        velocity (pygame.Vector2): The projectile's velocity vector.
        size (int): The radius of the projectile's circular body.
        color (tuple[int, int, int]): The display color of the projectile.
        lifespan (int): The number of frames the projectile will exist for.
        owner (Unit): The unit that fired the projectile.
    """
    def __init__(self, x, y, angle, owner):
        """Initializes a Projectile instance.

        Args:
            x (float): The initial x-coordinate.
            y (float): The initial y-coordinate.
            angle (float): The initial angle of travel in radians.
            owner (Unit): The unit that fired this projectile.
        """
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(5, 0).rotate(np.rad2deg(angle))
        self.size = 3
        self.color = (255, 255, 0) # Yellow
        self.lifespan = 150 # Frames
        self.owner = owner

    def get_bounding_box(self):
        """Returns a Rectangle representing the projectile's bounding box.

        Returns:
            Rectangle: An axis-aligned bounding box for the projectile.
        """
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def update(self):
        """Updates the projectile's position and decreases its lifespan."""
        self.position += self.velocity
        self.lifespan -= 1

    def draw(self, screen):
        """Draws the projectile on the screen.

        Args:
            screen (pygame.Surface): The pygame surface to draw on.
        """
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

# Helper functions for geometry have been moved to math_utils.py
