import unittest
import pygame

class TestSimulationLogic(unittest.TestCase):

    def test_collision(self):
        """
        Tests that pygame.Rect.colliderect() correctly identifies collisions.
        """
        rect1 = pygame.Rect(100, 100, 50, 50)
        rect2 = pygame.Rect(120, 120, 50, 50)
        self.assertTrue(rect1.colliderect(rect2), "Rects should collide")

    def test_no_collision(self):
        """
        Tests that pygame.Rect.colliderect() correctly identifies a lack of collision.
        """
        rect1 = pygame.Rect(100, 100, 50, 50)
        rect2 = pygame.Rect(200, 200, 50, 50)
        self.assertFalse(rect1.colliderect(rect2), "Rects should not collide")

    def test_edge_collision(self):
        """
        Tests that pygame.Rect.colliderect() correctly identifies collisions at the edge.
        """
        rect1 = pygame.Rect(100, 100, 50, 50)
        rect2 = pygame.Rect(150, 100, 50, 50)
        self.assertFalse(rect1.colliderect(rect2), "Rects should not collide when touching at the edge")

        # Note: Pygame's colliderect considers touching rects as non-colliding if the width/height is positive.
        # A collision happens when pixels overlap. If rect1 is at x=100 with width=50, it occupies pixels 100-149.
        # If rect2 is at x=150, it occupies 150-199. They don't overlap.
        # To make them overlap by one pixel:
        rect3 = pygame.Rect(149, 100, 50, 50)
        self.assertTrue(rect1.colliderect(rect3), "Rects should collide when overlapping by one pixel")

if __name__ == '__main__':
    unittest.main()
