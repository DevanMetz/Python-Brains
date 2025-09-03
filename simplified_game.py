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
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.static_grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY.value, dtype=int)
        self.dynamic_grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY.value, dtype=int)

    def set_tile(self, grid_x, grid_y, tile_type):
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            self.static_grid[grid_x, grid_y] = tile_type.value

    def get_tile_value(self, grid_x, grid_y):
        """Gets the integer value of a tile at a given grid coordinate."""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            if self.dynamic_grid[grid_x, grid_y] != Tile.EMPTY.value:
                return self.dynamic_grid[grid_x, grid_y]
            return self.static_grid[grid_x, grid_y]
        return Tile.WALL.value # Out of bounds is a wall

    def update_dynamic_grid(self, units, target):
        """Clears and rebuilds the dynamic grid with current unit and target positions."""
        self.dynamic_grid.fill(Tile.EMPTY.value)
        if target:
            self.dynamic_grid[target[0], target[1]] = Tile.TARGET.value
        for unit in units:
            if 0 <= unit.x < self.grid_width and 0 <= unit.y < self.grid_height:
                self.dynamic_grid[unit.x, unit.y] = Tile.UNIT.value

class SimplifiedUnit:
    """Represents a single unit in the simplified simulation."""
    def __init__(self, id, x, y, brain):
        self.id = id
        self.x = x
        self.y = y
        self.brain = brain

    def clone(self):
        """Creates a deep copy of the unit."""
        cloned_brain = self.brain.clone()
        return SimplifiedUnit(self.id, self.x, self.y, cloned_brain)

class SimplifiedGame:
    """Manages the overall state of the simplified simulation."""
    def __init__(self, width=40, height=30, population_size=50, mlp_arch_str="16", perception_radius=1, steps_per_gen=100):
        self.tile_map = TileMap(width, height)
        self.units = []
        self.target = (width - 5, height // 2)
        self.population_size = population_size
        self.generation = 0

        self.perception_radius = perception_radius
        self.steps_per_generation = steps_per_gen

        # Build MLP architecture from string
        try:
            hidden_layers = [int(n.strip()) for n in mlp_arch_str.split(',') if n.strip()]
        except ValueError:
            print(f"Warning: Invalid MLP architecture string '{mlp_arch_str}'. Using default [16].")
            hidden_layers = [16]

        num_inputs = (self.perception_radius * 2 + 1)**2 - 1 + 2 # neighbors + target vector
        num_outputs = len(Action)
        self.mlp_arch = [num_inputs] + hidden_layers + [num_outputs]

        self._initialize_population()
        self._create_walls()

    def _initialize_population(self):
        """Creates the initial population of units."""
        self.units = []
        start_x = 5
        start_y = self.tile_map.grid_height // 2
        for i in range(self.population_size):
            brain = MLP(self.mlp_arch)
            self.units.append(SimplifiedUnit(id=i, x=start_x, y=start_y, brain=brain))
        self.tile_map.update_dynamic_grid(self.units, self.target)

    def _create_walls(self):
        """Creates a simple pattern of walls on the map."""
        if self.tile_map.grid_width > 20:
            for y in range(5, self.tile_map.grid_height - 5):
                self.tile_map.set_tile(20, y, Tile.WALL)
        if self.tile_map.grid_width > 25:
            for x in range(25, self.tile_map.grid_width - 5):
                self.tile_map.set_tile(x, 10, Tile.WALL)

    def update_simulation_with_results(self, results):
        """Updates the positions of units based on multiprocessing results."""
        for unit_id, new_x, new_y in results:
            # This can cause an index out of bounds if the unit list has changed
            if unit_id < len(self.units):
                self.units[unit_id].x = new_x
                self.units[unit_id].y = new_y
        self.tile_map.update_dynamic_grid(self.units, self.target)

    def evolve_population(self):
        """Evolves the population based on fitness (proximity to target)."""
        fitness_scores = []
        for unit in self.units:
            dist_sq = (unit.x - self.target[0])**2 + (unit.y - self.target[1])**2
            fitness_scores.append(1 / (dist_sq + 1))

        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_units = [self.units[i] for i in sorted_indices]

        next_gen_units = []
        num_elites = self.population_size // 10
        start_x = 5
        start_y = self.tile_map.grid_height // 2
        for i in range(num_elites):
            elite_unit = sorted_units[i].clone()
            elite_unit.x = start_x
            elite_unit.y = start_y
            next_gen_units.append(elite_unit)

        while len(next_gen_units) < self.population_size:
            parent1 = np.random.choice(sorted_units[:self.population_size//2])
            parent2 = np.random.choice(sorted_units[:self.population_size//2])
            child_brain = MLP.crossover(parent1.brain, parent2.brain)
            child_brain.mutate(mutation_rate=0.05, mutation_amount=0.1)
            next_gen_units.append(SimplifiedUnit(id=len(next_gen_units), x=start_x, y=start_y, brain=child_brain))

        self.units = next_gen_units
        for i, unit in enumerate(self.units):
            unit.id = i

        self.generation += 1
        self.tile_map.update_dynamic_grid(self.units, self.target)
        print(f"Generation {self.generation} complete. Best fitness: {max(fitness_scores):.4f}")

# --- Testable Helper Functions ---

def determine_new_position(unit_x, unit_y, action, static_grid, dynamic_grid):
    """
    Calculates a unit's new position based on an action and collision rules.
    This is a pure function for easy testing.
    """
    new_x, new_y = unit_x, unit_y
    if action == Action.MOVE_N: new_y -= 1
    elif action == Action.MOVE_E: new_x += 1
    elif action == Action.MOVE_S: new_y += 1
    elif action == Action.MOVE_W: new_x -= 1

    grid_width, grid_height = static_grid.shape
    final_x, final_y = unit_x, unit_y

    if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
        is_wall = static_grid[new_x, new_y] == Tile.WALL.value
        if not is_wall:
            final_x, final_y = new_x, new_y

    return final_x, final_y

# --- Multiprocessing Function ---

def process_unit_logic(args):
    """
    Runs a single simulation step for one unit.
    This is a top-level function to be called by a multiprocessing pool.
    """
    unit_id, unit_x, unit_y, brain_weights, brain_biases, static_grid, dynamic_grid, target_pos, mlp_arch, perception_radius = args

    brain = MLP(mlp_arch)
    brain.weights = brain_weights
    brain.biases = brain_biases

    num_neighbors = (perception_radius * 2 + 1)**2 - 1
    inputs = np.zeros(num_neighbors + 2)
    idx = 0
    grid_width, grid_height = static_grid.shape

    for dy in range(-perception_radius, perception_radius + 1):
        for dx in range(-perception_radius, perception_radius + 1):
            if dx == 0 and dy == 0: continue
            px, py = unit_x + dx, unit_y + dy
            tile_val = Tile.WALL.value
            if 0 <= px < grid_width and 0 <= py < grid_height:
                tile_val = dynamic_grid[px, py] if dynamic_grid[px, py] != Tile.EMPTY.value else static_grid[px, py]
            inputs[idx] = tile_val
            idx += 1

    dx_to_target = (target_pos[0] - unit_x) / grid_width
    dy_to_target = (target_pos[1] - unit_y) / grid_height
    inputs[num_neighbors] = dx_to_target
    inputs[num_neighbors + 1] = dy_to_target

    action_probs = brain.forward(inputs)
    action = Action(np.argmax(action_probs))

    final_x, final_y = determine_new_position(unit_x, unit_y, action, static_grid, dynamic_grid)

    return (unit_id, final_x, final_y)
