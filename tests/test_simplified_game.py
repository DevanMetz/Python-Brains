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

def test_get_vision_inputs_correct_normalization():
    """Tests the 8-directional range-limited vision system."""
    static_grid = np.full((20, 20), Tile.EMPTY.value, dtype=int)
    # Wall at y=7
    static_grid[:, 7] = Tile.WALL.value

    start_x, start_y = 10, 10
    vision_range = 5

    vision = get_vision_inputs(start_x, start_y, static_grid, vision_range)

    # N (index 0): Wall is at y=7, so distance is 3.
    # Expected value: 3 / 5 = 0.6
    assert np.isclose(vision[0], 3.0 / vision_range)

def test_get_vision_inputs_no_wall_in_range():
    """Tests that vision returns 1.0 when no wall is found within range."""
    static_grid = np.full((20, 20), Tile.EMPTY.value, dtype=int)
    start_x, start_y = 10, 10
    vision_range = 5

    vision = get_vision_inputs(start_x, start_y, static_grid, vision_range)

    # All directions should be 1.0 as no wall is in range
    assert np.allclose(vision, 1.0)

def test_get_vision_inputs_wall_at_max_range():
    """Tests vision when a wall is exactly at the maximum vision range."""
    static_grid = np.full((20, 20), Tile.EMPTY.value, dtype=int)
    static_grid[10, 5] = Tile.WALL.value # Wall at y=5, dist=5
    start_x, start_y = 10, 10
    vision_range = 5

    vision = get_vision_inputs(start_x, start_y, static_grid, vision_range)

    # N (index 0): Wall is at y=5, so distance is 5.
    # Expected value: 5 / 5 = 1.0
    assert np.isclose(vision[0], 1.0)
