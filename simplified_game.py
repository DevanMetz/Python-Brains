"""
This file contains the core logic for the simplified, tile-based simulation.
It is designed to be independent of Pygame for easier testing and multiprocessing.
"""
import numpy as np
from enum import Enum
from mlp import MLP

# --- Enums ---
class Tile(Enum):
    """An enumeration for the different types of tiles."""
    EMPTY = 0
    WALL = 1
    UNIT = 2
    TARGET = 3

class Action(Enum):
    """Discrete actions a unit can take."""
    MOVE_N = 0
    MOVE_E = 1
    MOVE_S = 2
    MOVE_W = 3
    STAY = 4

# --- Simulation Classes ---
class TileMap:
    """Manages the tile-based map for the simulation."""
    def __init__(self, grid_width, grid_height, static_grid=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        if static_grid is not None:
            self.static_grid = static_grid
        else:
            self.static_grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY.value, dtype=int)
        self.dynamic_grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY.value, dtype=int)

    def set_tile(self, grid_x, grid_y, tile_type):
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.static_grid[grid_x, grid_y] = tile_type.value

    def get_tile_value(self, grid_x, grid_y):
        """Gets the integer value of a tile at a given grid coordinate."""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.static_grid[grid_x, grid_y]
        return Tile.WALL.value

    def update_dynamic_grid(self, units, target):
        """Clears and rebuilds the dynamic grid with current unit and target positions."""
        self.dynamic_grid.fill(Tile.EMPTY.value)
        if target: self.dynamic_grid[target[0], target[1]] = Tile.TARGET.value
        for unit in units:
            if 0 <= unit.x < self.grid_width and 0 <= unit.y < self.grid_height:
                self.dynamic_grid[unit.x, unit.y] = Tile.UNIT.value

class SimplifiedUnit:
    """Represents a single unit in the simplified simulation."""
    def __init__(self, id, x, y, brain):
        self.id, self.x, self.y, self.brain = id, x, y, brain

    def clone(self):
        return SimplifiedUnit(self.id, self.x, self.y, self.brain.clone())

class SimplifiedGame:
    """Manages the overall state of the simplified simulation."""
    def __init__(self, width=40, height=30, population_size=100, mlp_arch_str="16",
                 steps_per_gen=100, mutation_rate=0.05, static_grid=None):
        self.tile_map = TileMap(width, height, static_grid)
        self.units = []
        self.target = (width - 5, height // 2)
        self.population_size = population_size
        self.generation = 0
        self.fittest_brain = None

        self.steps_per_generation = steps_per_gen
        self.mutation_rate = mutation_rate

        try:
            hidden_layers = [int(n.strip()) for n in mlp_arch_str.split(',') if n.strip()]
        except ValueError: hidden_layers = [16]

        num_inputs = 10 # 8 vision rays + 2 for target vector
        num_outputs = len(Action)
        self.mlp_arch = [num_inputs] + hidden_layers + [num_outputs]

        self._initialize_population()
        if static_grid is None: self._create_walls()

    def _initialize_population(self):
        self.units = []
        start_x, start_y = 5, self.tile_map.grid_height // 2
        for i in range(self.population_size):
            self.units.append(SimplifiedUnit(i, start_x, start_y, MLP(self.mlp_arch)))
        self.tile_map.update_dynamic_grid(self.units, self.target)

    def _create_walls(self):
        if self.tile_map.grid_width > 20:
            for y in range(5, self.tile_map.grid_height - 5): self.tile_map.set_tile(20, y, Tile.WALL)
        if self.tile_map.grid_width > 25:
            for x in range(25, self.tile_map.grid_width - 5): self.tile_map.set_tile(x, 10, Tile.WALL)

    def update_simulation_with_results(self, results):
        for unit_id, new_x, new_y in results:
            if unit_id < len(self.units):
                self.units[unit_id].x, self.units[unit_id].y = new_x, new_y
        self.tile_map.update_dynamic_grid(self.units, self.target)

    def evolve_population(self):
        fitness_scores = [(1 / ((u.x - self.target[0])**2 + (u.y - self.target[1])**2 + 1)) for u in self.units]
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_units = [self.units[i] for i in sorted_indices]
        self.fittest_brain = sorted_units[0].brain.clone()
        num_elites = self.population_size // 10
        start_x, start_y = 5, self.tile_map.grid_height // 2
        next_gen_units = [u.clone() for u in sorted_units[:num_elites]]
        for unit in next_gen_units: unit.x, unit.y = start_x, start_y
        elite_pool = sorted_units[:num_elites] if num_elites > 0 else [sorted_units[0]]
        while len(next_gen_units) < self.population_size:
            parent = np.random.choice(elite_pool)
            child_brain = parent.brain.clone()
            child_brain.mutate(mutation_rate=self.mutation_rate, mutation_amount=0.1)
            next_gen_units.append(SimplifiedUnit(len(next_gen_units), start_x, start_y, child_brain))
        self.units = next_gen_units
        for i, unit in enumerate(self.units): unit.id = i
        self.generation += 1
        self.tile_map.update_dynamic_grid(self.units, self.target)
        print(f"Generation {self.generation} complete. Best fitness: {max(fitness_scores):.4f}")

def get_vision_inputs(start_x, start_y, static_grid):
    """Casts 8 rays to find distance to the nearest wall."""
    directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    vision_inputs = np.zeros(8)
    grid_width, grid_height = static_grid.shape
    max_dist = max(grid_width, grid_height)

    for i, (dx, dy) in enumerate(directions):
        dist = 1
        while True:
            check_x, check_y = start_x + dx * dist, start_y + dy * dist
            if not (0 <= check_x < grid_width and 0 <= check_y < grid_height):
                vision_inputs[i] = 1.0 # Hit edge of map
                break
            if static_grid[check_x, check_y] == Tile.WALL.value:
                vision_inputs[i] = 1.0 - (dist / max_dist)
                break
            dist += 1
    return vision_inputs

def determine_new_position(unit_x, unit_y, action, static_grid):
    new_x, new_y = unit_x, unit_y
    if action == Action.MOVE_N: new_y -= 1
    elif action == Action.MOVE_E: new_x += 1
    elif action == Action.MOVE_S: new_y += 1
    elif action == Action.MOVE_W: new_x -= 1
    grid_width, grid_height = static_grid.shape
    final_x, final_y = unit_x, unit_y
    if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
        if static_grid[new_x, new_y] != Tile.WALL.value:
            final_x, final_y = new_x, new_y
    return final_x, final_y

def process_unit_logic(args):
    unit_id, unit_x, unit_y, brain_weights, brain_biases, static_grid, target_pos, mlp_arch = args
    brain = MLP(mlp_arch)
    brain.weights, brain.biases = brain_weights, brain_biases

    vision_inputs = get_vision_inputs(unit_x, unit_y, static_grid)

    dx_to_target = (target_pos[0] - unit_x) / static_grid.shape[0]
    dy_to_target = (target_pos[1] - unit_y) / static_grid.shape[1]
    target_inputs = np.array([dx_to_target, dy_to_target])

    inputs = np.concatenate((vision_inputs, target_inputs))

    action_probs, _ = brain.forward(inputs) # We only need the final output here
    action = Action(np.argmax(action_probs))
    final_x, final_y = determine_new_position(unit_x, unit_y, action, static_grid)
    return (unit_id, final_x, final_y)
