import numpy as np
import pygame
from mlp import MLP
from quadtree import Rectangle

from map import Tile

class Unit:
    """
    Represents a single unit in the game, controlled by an MLP brain.
    """
    def __init__(self, id, x, y, tile_map, brain=None, num_whiskers=7, whisker_length=150, perceivable_types=None):
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
        """
        Serializes the unit's state to a dictionary for multiprocessing.
        Note: The brain is passed directly as it's picklable (thanks to numpy).
        The tile_map is not included as it's handled globally in worker processes.
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
        """Returns a Rectangle representing the unit's bounding box."""
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def get_inputs(self, world_objects, target):
        """
        Calculates inputs for the MLP brain, including whisker data and target vector.
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
                dist = line_circle_intersection(start_point, end_point, obj.position, obj.size)
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
        """Creates and returns a new projectile if cooldown is over."""
        if self.attack_cooldown <= 0:
            self.attack_cooldown = self.max_cooldown
            return Projectile(self.position.x, self.position.y, self.angle, self)
        return None

    def _check_wall_collision(self, position):
        """Checks if a position collides with a wall tile."""
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
        """
        Updates the unit's state based on MLP actions, including wall collision and sliding.
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
    """Represents a simple rectangular wall."""
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (100, 100, 100)
        self.type = "wall"

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Target:
    """Represents the target the unit should seek."""
    def __init__(self, x, y):
        self.id = -1 # Static ID for the target
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (0, 255, 0)
        self.type = "target"

    def to_dict(self):
        return {
            "id": self.id,
            "position": (self.position.x, self.position.y),
            "size": self.size,
            "type": self.type,
        }

    def get_bounding_box(self):
        """Returns a Rectangle representing the target's bounding box."""
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

class Enemy:
    """Represents a stationary enemy with health."""
    def __init__(self, x, y):
        self.id = -2 # Static ID for the enemy
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (255, 0, 0)
        self.health = 100
        self.type = "enemy"

    def to_dict(self):
        return {
            "id": self.id,
            "position": (self.position.x, self.position.y),
            "size": self.size,
            "type": self.type,
            "health": self.health,
        }

    def get_bounding_box(self):
        """Returns a Rectangle representing the enemy's bounding box."""
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)
        # Draw health bar
        if self.health < 100:
            bar_width = self.size * 2
            bar_height = 5
            health_frac = self.health / 100.0
            pygame.draw.rect(screen, (255,0,0), (self.position.x - self.size, self.position.y - self.size - 10, bar_width, bar_height))
            pygame.draw.rect(screen, (0,255,0), (self.position.x - self.size, self.position.y - self.size - 10, bar_width * health_frac, bar_height))

class Projectile:
    """Represents a projectile fired by a unit."""
    def __init__(self, x, y, angle, owner):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(5, 0).rotate(np.rad2deg(angle))
        self.size = 3
        self.color = (255, 255, 0) # Yellow
        self.lifespan = 150 # Frames
        self.owner = owner

    def get_bounding_box(self):
        """Returns a Rectangle representing the projectile's bounding box."""
        return Rectangle(self.position.x, self.position.y, self.size * 2, self.size * 2)

    def update(self):
        self.position += self.velocity
        self.lifespan -= 1

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

# Helper functions for geometry

def line_intersection(p0, p1, p2, p3):
    """
    Checks if line segment 'p0p1' and 'p2p3' intersect.
    Returns the intersection point or None.
    p0, p1, p2, p3 are tuples or pygame.Vector2
    """
    s1_x = p1[0] - p0[0]
    s1_y = p1[1] - p0[1]
    s2_x = p3[0] - p2[0]
    s2_y = p3[1] - p2[1]

    denominator = (-s2_x * s1_y + s1_x * s2_y)

    if denominator == 0:
        return None # Lines are parallel

    s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / denominator
    t = ( s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / denominator

    if 0 <= s <= 1 and 0 <= t <= 1:
        # Intersection detected
        return pygame.Vector2(p0[0] + (t * s1_x), p0[1] + (t * s1_y))
    return None

def line_circle_intersection(p1, p2, circle_center, radius):
    """
    Calculates the intersection of a line segment and a circle.
    Returns the distance from p1 to the closest intersection point, or None.
    """
    p1 = pygame.Vector2(p1)
    p2 = pygame.Vector2(p2)
    circle_center = pygame.Vector2(circle_center)

    d = p2 - p1
    f = p1 - circle_center

    a = d.dot(d)
    b = 2 * f.dot(d)
    c = f.dot(f) - radius * radius

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    else:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        # Check if the intersection points are on the line segment
        if 0 <= t1 <= 1:
            # t1 is an intersection on the segment
            return p1.distance_to(p1 + t1 * d)
        # We don't care about t2 if t1 is valid, as t1 will be closer
        return None
