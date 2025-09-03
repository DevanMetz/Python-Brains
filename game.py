import numpy as np
import pygame
from mlp import MLP

class Unit:
    """
    Represents a single unit in the game, controlled by an MLP brain.
    """
    def __init__(self, x, y, brain=None):
        self.position = pygame.Vector2(x, y)
        self.angle = np.random.uniform(0, 2 * np.pi) # Facing direction in radians
        self.velocity = pygame.Vector2(0, 0)
        self.size = 10 # Radius for drawing and collision
        self.color = (0, 150, 255) # Blue
        self.speed = 2.0

        self.brain = brain
        self.whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, 7) # 7 whiskers in a 180-degree cone
        self.whisker_length = 150

    def get_inputs(self, world_objects):
        """
        Calculates the inputs for the MLP brain using the whisker system.
        Each whisker returns a value indicating the proximity of a detected object.
        """
        inputs = []

        for whisker_angle in self.whisker_angles:
            abs_angle = self.angle + whisker_angle
            start_point = self.position
            end_point = self.position + pygame.Vector2(self.whisker_length, 0).rotate(np.rad2deg(abs_angle))

            closest_dist = self.whisker_length

            for obj in world_objects:
                if obj is self:
                    continue

                if isinstance(obj, Wall):
                    # Check for intersection with each of the wall's four sides
                    points = [
                        (obj.rect.left, obj.rect.top), (obj.rect.right, obj.rect.top),
                        (obj.rect.right, obj.rect.bottom), (obj.rect.left, obj.rect.bottom)
                    ]
                    for i in range(4):
                        p1 = points[i]
                        p2 = points[(i + 1) % 4]
                        intersect_point = line_intersection(start_point, end_point, p1, p2)
                        if intersect_point:
                            dist = self.position.distance_to(intersect_point)
                            if dist < closest_dist:
                                closest_dist = dist
                elif isinstance(obj, (Unit, Target)):
                    # Check for line-circle intersection
                    intersect_dist = line_circle_intersection(start_point, end_point, obj.position, obj.size)
                    if intersect_dist is not None:
                        if intersect_dist < closest_dist:
                            closest_dist = intersect_dist

            # The input is the inverse of the distance (closer is a stronger signal)
            inputs.append(1.0 - (closest_dist / self.whisker_length))

        # Also add unit's own velocity and angle as inputs
        inputs.append(self.velocity.length() / self.speed)
        inputs.append(self.angle / (2 * np.pi))

        return np.array(inputs)

    def update(self, actions):
        """
        Updates the unit's state based on the MLP's output actions.
        """
        # Actions are expected to be in the range [-1, 1] from the tanh activation
        turn_action = actions[0][0] # Turn left/right
        move_action = actions[0][1] # Move forward/backward

        # Update angle
        self.angle += turn_action * 0.1 # Adjust sensitivity as needed

        # Update velocity based on move action
        # We use max(0, ...) to prevent moving backward
        forward_speed = max(0, move_action) * self.speed
        self.velocity = pygame.Vector2(forward_speed, 0).rotate(np.rad2deg(self.angle))

        # Update position
        self.position += self.velocity

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

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

class Target:
    """Represents the target the unit should seek."""
    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.size = 15
        self.color = (0, 255, 0) # Green

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
