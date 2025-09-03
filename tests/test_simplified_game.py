import pytest
import numpy as np
from simplified_game import SimplifiedGame, Tile, Action, determine_new_position, get_vision_inputs
from mlp import MLP

@pytest.fixture
def game():
    """Fixture to create a default SimplifiedGame instance for testing."""
    return SimplifiedGame(width=20, height=15)

def test_game_initialization(game):
    """Tests that the game and its components are initialized correctly."""
    assert game.tile_map.grid_width == 20
    assert game.tile_map.grid_height == 15
    assert game.mlp_arch[0] == 10 # 8 vision + 2 target
    assert game.mlp_arch[-1] == 5 # 5 actions

def test_determine_new_position_can_move_to_empty():
    """Tests that a unit can move into an empty space."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    start_x, start_y = 5, 5

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_E, static_grid)
    assert (final_x, final_y) == (6, 5)

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_N, static_grid)
    assert (final_x, final_y) == (5, 4)

def test_determine_new_position_cannot_move_into_wall():
    """Tests that a unit cannot move into a wall."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    static_grid[6, 5] = Tile.WALL.value
    start_x, start_y = 5, 5

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_E, static_grid)
    assert (final_x, final_y) == (start_x, start_y)

def test_get_vision_inputs():
    """Tests the 8-directional ray-casting vision system."""
    static_grid = np.full((20, 20), Tile.EMPTY.value, dtype=int)
    # Create a wall box around (10, 10)
    # Top wall at y=7
    static_grid[:, 7] = Tile.WALL.value
    # Right wall at x=13
    static_grid[13, :] = Tile.WALL.value

    start_x, start_y = 10, 10

    vision = get_vision_inputs(start_x, start_y, static_grid)

    # Expected distances
    # N: (10,10) to (10,7) -> dist = 3
    # NE: (10,10) to (13,7) -> dist = 3
    # E: (10,10) to (13,10) -> dist = 3

    max_dist = max(static_grid.shape)

    # N (index 0)
    assert np.isclose(vision[0], 1.0 - (3 / max_dist))
    # NE (index 1)
    assert np.isclose(vision[1], 1.0 - (3 / max_dist))
    # E (index 2)
    assert np.isclose(vision[2], 1.0 - (3 / max_dist))

    # Test hitting the edge of the map
    # Create a new empty map
    static_grid_empty = np.full((20, 20), Tile.EMPTY.value, dtype=int)
    start_x_edge, start_y_edge = 0, 0
    vision_edge = get_vision_inputs(start_x_edge, start_y_edge, static_grid_empty)

    # W, NW, N should all be 1.0 (hit edge immediately)
    assert np.isclose(vision_edge[0], 1.0) # N
    assert np.isclose(vision_edge[6], 1.0) # W
    assert np.isclose(vision_edge[7], 1.0) # NW
