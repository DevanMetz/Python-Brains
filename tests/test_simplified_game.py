import pytest
import numpy as np
from simplified_game import SimplifiedGame, SimplifiedUnit, Tile, Action, determine_new_position
from mlp import MLP

@pytest.fixture
def game():
    """Fixture to create a default SimplifiedGame instance for testing."""
    return SimplifiedGame(width=20, height=15, population_size=10)

def test_game_initialization(game):
    """Tests that the game and its components are initialized correctly."""
    assert game.tile_map.grid_width == 20
    assert game.tile_map.grid_height == 15
    assert len(game.units) == 10
    assert game.target is not None

def test_determine_new_position_can_move_to_empty():
    """Tests that a unit can move into an empty space."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    dynamic_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)

    start_x, start_y = 5, 5

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_E, static_grid, dynamic_grid)
    assert (final_x, final_y) == (6, 5)

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_N, static_grid, dynamic_grid)
    assert (final_x, final_y) == (5, 4)

def test_determine_new_position_cannot_move_into_wall():
    """Tests that a unit cannot move into a wall."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    static_grid[6, 5] = Tile.WALL.value
    dynamic_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)

    start_x, start_y = 5, 5

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_E, static_grid, dynamic_grid)
    assert (final_x, final_y) == (start_x, start_y)

def test_determine_new_position_can_move_into_unit():
    """Tests that a unit CAN move into a tile occupied by another unit."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    dynamic_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    dynamic_grid[6, 5] = Tile.UNIT.value # Another unit is here

    start_x, start_y = 5, 5

    # The new logic ignores other units, so the move should be successful
    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_E, static_grid, dynamic_grid)
    assert (final_x, final_y) == (6, 5)

def test_determine_new_position_stay_action():
    """Tests that the STAY action results in no movement."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    dynamic_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)

    start_x, start_y = 5, 5

    final_x, final_y = determine_new_position(start_x, start_y, Action.STAY, static_grid, dynamic_grid)
    assert (final_x, final_y) == (start_x, start_y)

def test_determine_new_position_cannot_move_off_map():
    """Tests that a unit cannot move outside the map boundaries."""
    static_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)
    dynamic_grid = np.full((10, 10), Tile.EMPTY.value, dtype=int)

    start_x, start_y = 0, 0

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_W, static_grid, dynamic_grid)
    assert (final_x, final_y) == (start_x, start_y)

    final_x, final_y = determine_new_position(start_x, start_y, Action.MOVE_N, static_grid, dynamic_grid)
    assert (final_x, final_y) == (start_x, start_y)
