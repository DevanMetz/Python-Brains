import unittest
import numpy as np
import sys
import os
import pygame

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game import Unit, Target, Enemy, Projectile
from map import Tile, TileMap

class TestGameObjects(unittest.TestCase):
    def setUp(self):
        """Set up common resources for tests."""
        pygame.init()
        # Use pixel dimensions that are a multiple of the tile size
        self.tile_map = TileMap(width=800, height=800, tile_size=40)

    def tearDown(self):
        """Clean up resources after tests."""
        pygame.quit()

    def test_unit_initialization(self):
        """Test that a Unit is initialized with the correct attributes."""
        unit = Unit(0, 50, 50, self.tile_map, num_whiskers=5, whisker_length=100)
        self.assertEqual(unit.id, 0)
        self.assertEqual(unit.position, pygame.Vector2(50, 50))
        self.assertEqual(unit.num_whiskers, 5)
        self.assertEqual(unit.whisker_length, 100)
        self.assertIsNone(unit.brain)

    def test_unit_to_dict_serialization(self):
        """Test the serialization of a Unit object to a dictionary."""
        brain_mock = "mock_brain" # A simple string or any picklable object
        unit = Unit(1, 100, 150, self.tile_map, brain=brain_mock, num_whiskers=3)
        unit.angle = np.pi / 2
        unit.velocity = pygame.Vector2(1, 2)

        unit_dict = unit.to_dict()

        self.assertEqual(unit_dict['id'], 1)
        self.assertEqual(unit_dict['position'], (100, 150))
        self.assertAlmostEqual(unit_dict['angle'], np.pi / 2)
        self.assertEqual(unit_dict['velocity'], (1, 2))
        self.assertEqual(unit_dict['brain'], "mock_brain")
        self.assertEqual(unit_dict['num_whiskers'], 3)

    def test_unit_get_bounding_box(self):
        """Test the bounding box calculation for a Unit."""
        unit = Unit(0, 100, 100, self.tile_map)
        unit.size = 10
        bbox = unit.get_bounding_box()
        self.assertEqual(bbox.x, 100)
        self.assertEqual(bbox.y, 100)
        self.assertEqual(bbox.w, 20)
        self.assertEqual(bbox.h, 20)

    def test_unit_wall_collision(self):
        """Test the wall collision check."""
        # Set a wall tile at a specific grid location
        self.tile_map.set_tile(5, 5, Tile.WALL)

        # Position the unit in the center of the wall tile
        unit_pos_in_wall = pygame.Vector2(5 * 40 + 20, 5 * 40 + 20)
        unit = Unit(0, unit_pos_in_wall.x, unit_pos_in_wall.y, self.tile_map)

        # This position should be inside a wall
        self.assertTrue(unit._check_wall_collision(unit.position))

        # This position should be clear of walls
        unit.position = pygame.Vector2(100, 100)
        self.assertFalse(unit._check_wall_collision(unit.position))

    def test_unit_update_movement(self):
        """Test the unit's movement logic from the update method."""
        unit = Unit(0, 100, 100, self.tile_map)
        unit.speed = 1.0

        # Action: Move forward, turn right slightly
        actions = np.array([[0.1, 1.0]]) # turn, move

        initial_angle = unit.angle
        unit.update(actions, [])

        self.assertNotEqual(unit.position, pygame.Vector2(100, 100))
        # The turn action is scaled by 0.1 in the update method
        self.assertAlmostEqual(unit.angle, initial_angle + 0.01)

    def test_unit_update_attack(self):
        """Test the unit's attack logic from the update method."""
        unit = Unit(0, 100, 100, self.tile_map)

        # Action: Attack
        actions = np.array([[0.0, 0.0, 1.0]]) # turn, move, attack

        projectiles = []
        unit.update(actions, projectiles)

        self.assertEqual(len(projectiles), 1)
        self.assertIsInstance(projectiles[0], Projectile)
        self.assertEqual(projectiles[0].owner, unit)

    def test_unit_attack_cooldown(self):
        """Test the cooldown mechanism of the attack action."""
        unit = Unit(0, 100, 100, self.tile_map)
        unit.max_cooldown = 5

        # First attack should succeed
        projectile = unit.attack()
        self.assertIsNotNone(projectile)
        self.assertEqual(unit.attack_cooldown, 5)

        # Immediate second attack should fail due to cooldown
        projectile = unit.attack()
        self.assertIsNone(projectile)

        # Simulate a few frames passing
        unit.attack_cooldown -= 3

        # Attack should still fail
        projectile = unit.attack()
        self.assertIsNone(projectile)

        # Cooldown finishes
        unit.attack_cooldown = 0

        # Attack should succeed again
        projectile = unit.attack()
        self.assertIsNotNone(projectile)

    def test_target_initialization(self):
        """Test that a Target is initialized correctly."""
        target = Target(200, 300)
        self.assertEqual(target.position, pygame.Vector2(200, 300))
        self.assertEqual(target.type, "target")

    def test_enemy_initialization(self):
        """Test that an Enemy is initialized correctly."""
        enemy = Enemy(400, 500)
        self.assertEqual(enemy.position, pygame.Vector2(400, 500))
        self.assertEqual(enemy.type, "enemy")
        self.assertEqual(enemy.health, 100)

    def test_projectile_initialization(self):
        """Test that a Projectile is initialized correctly."""
        owner_unit = Unit(0, 0, 0, self.tile_map)
        projectile = Projectile(10, 20, np.pi / 4, owner_unit)
        self.assertEqual(projectile.position, pygame.Vector2(10, 20))
        self.assertEqual(projectile.owner, owner_unit)

if __name__ == '__main__':
    unittest.main()
