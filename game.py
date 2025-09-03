import numpy as np
import pygame
from mlp import MLP

class Unit:
    """
    Represents a single unit in the game, controlled by an MLP brain.
    """
    def __init__(self, x, y, brain=None, num_whiskers=7, perceivable_types=None):
        self.position = pygame.Vector2(x, y)
        self.angle = np.random.uniform(0, 2 * np.pi) # Facing direction in radians
        self.velocity = pygame.Vector2(0, 0)
        self.size = 10 # Radius for drawing and collision
        self.color = (0, 150, 255) # Blue
        self.speed = 2.0
        self.type = "unit"
        self.damage_dealt = 0

        self.brain = brain
        self.num_whiskers = num_whiskers
        self.perceivable_types = perceivable_types if perceivable_types is not None else ["wall", "enemy", "target", "unit"]

        self.attack_cooldown = 0
        self.max_cooldown = 30 # Can fire every 30 frames

        if self.num_whiskers > 1:
            self.whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_whiskers)
        else:
            self.whisker_angles = np.array([0]) # Single whisker facing forward
        self.whisker_length = 150

    def get_inputs(self, world_objects):
        """
        Calculates inputs for the MLP brain using typed whisker perception.
        For each whisker, it generates an input for each perceivable object type.
        """
        num_perceivables = len(self.perceivable_types)
        # Create a matrix for whisker inputs: rows=whiskers, cols=perceivable_types
        whisker_inputs = np.zeros((self.num_whiskers, num_perceivables))

        for i, whisker_angle in enumerate(self.whisker_angles):
            abs_angle = self.angle + whisker_angle
            start_point = self.position
            end_point = self.position + pygame.Vector2(self.whisker_length, 0).rotate(np.rad2deg(abs_angle))

            closest_dist = self.whisker_length
            detected_type = None

            # Find the closest object intersected by this whisker
            for obj in world_objects:
                if obj is self:
                    continue

                dist = self.whisker_length + 1
                if isinstance(obj, Wall):
                    points = [
                        (obj.rect.left, obj.rect.top), (obj.rect.right, obj.rect.top),
                        (obj.rect.right, obj.rect.bottom), (obj.rect.left, obj.rect.bottom)
                    ]
                    for j in range(4):
                        p1, p2 = points[j], points[(j + 1) % 4]
                        intersect_point = line_intersection(start_point, end_point, p1, p2)
                        if intersect_point:
                            dist = self.position.distance_to(intersect_point)
                elif hasattr(obj, 'position'): # Circle-based objects
                    intersect_dist = line_circle_intersection(start_point, end_point, obj.position, obj.size)
                    if intersect_dist is not None:
                        dist = intersect_dist

                if dist < closest_dist:
                    closest_dist = dist
                    detected_type = obj.type

            # If an object was detected, update the corresponding input neuron
            if detected_type and detected_type in self.perceivable_types:
                type_index = self.perceivable_types.index(detected_type)
                whisker_inputs[i, type_index] = 1.0 - (closest_dist / self.whisker_length)

        # Flatten the whisker inputs and add other state info
        flat_whisker_inputs = whisker_inputs.flatten()
        other_inputs = np.array([self.velocity.length() / self.speed, self.angle / (2 * np.pi)])

        return np.concatenate((flat_whisker_inputs, other_inputs))

    def attack(self):
        """Creates and returns a new projectile if cooldown is over."""
        if self.attack_cooldown <= 0:
            self.attack_cooldown = self.max_cooldown
            return Projectile(self.position.x, self.position.y, self.angle, self)
        return None

    def update(self, actions, world_projectiles):
        """
        Updates the unit's state based on the MLP's output actions.
        """
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # Actions are expected to be in the range [-1, 1] from the tanh activation
        turn_action = actions[0][0]
        move_action = actions[0][1]

        self.angle += turn_action * 0.1
        forward_speed = max(0, move_action) * self.speed
        self.velocity = pygame.Vector2(forward_speed, 0).rotate(np.rad2deg(self.angle))
        self.position += self.velocity

        # Handle attack action if it exists
        if len(actions[0]) > 2:
            attack_action = actions[0][2]
            if attack_action > 0.5: # Threshold to fire
                projectile = self.attack()
                if projectile:
                    world_projectiles.append(projectile)

    def draw(self, screen):
        # Draw body
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

        # Draw direction indicator
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
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (0, 255, 0) # Green
        self.type = "target"

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), self.size)

class Enemy:
    """Represents a stationary enemy with health."""
    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (255, 0, 0) # Red
        self.health = 100
        self.type = "enemy"

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
