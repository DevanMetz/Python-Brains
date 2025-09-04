"""
This file contains the core logic for the simplified, tile-based simulation.
It is designed to be independent of Pygame for easier testing and multiprocessing.
"""
import numpy as np
from enum import Enum
from mlp import MLP
from mlp_batch_processor import MLPBatchProcessor, OPENCL_AVAILABLE

# --- Enums ---
class Tile(Enum):
    EMPTY = 0
    WALL = 1
    UNIT = 2
    TARGET = 3

class Action(Enum):
    MOVE_N = 0
    MOVE_E = 1
    MOVE_S = 2
    MOVE_W = 3
    STAY = 4

# --- Simulation Classes ---
class TileMap:
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
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.static_grid[grid_x, grid_y]
        return Tile.WALL.value

    def update_dynamic_grid(self, units, target):
        self.dynamic_grid.fill(Tile.EMPTY.value)
        if target: self.dynamic_grid[target[0], target[1]] = Tile.TARGET.value
        for unit in units:
            if 0 <= unit.x < self.grid_width and 0 <= unit.y < self.grid_height:
                self.dynamic_grid[unit.x, unit.y] = Tile.UNIT.value

class SimplifiedUnit:
    def __init__(self, id, x, y, brain: MLP):
        self.id, self.x, self.y, self.brain = id, x, y, brain
        self.visited_tiles = set([(x, y)])

    def clone(self):
        return SimplifiedUnit(self.id, self.x, self.y, self.brain.clone())

class SimplifiedGame:
    def __init__(self, width=40, height=30, population_size=100, mlp_arch_str="16",
                 perception_radius=5, steps_per_gen=100, mutation_rate=0.05,
                 exploration_bonus=0.0, static_grid=None):
        self.tile_map = TileMap(width, height, static_grid)
        self.units = []
        self.target = (width - 5, height // 2)
        self.population_size = population_size
        self.generation = 0
        self.fittest_brain = None
        self.batch_processor = None

        self.perception_radius = perception_radius
        self.steps_per_generation = steps_per_gen
        self.mutation_rate = mutation_rate
        self.exploration_bonus = exploration_bonus
        self.mlp_arch_str = mlp_arch_str

        self._setup_mlp_arch(mlp_arch_str)
        self._initialize_population()
        if static_grid is None: self._create_walls()

    def _setup_mlp_arch(self, mlp_arch_str):
        try:
            hidden_layers = [int(n.strip()) for n in mlp_arch_str.split(',') if n.strip()]
        except ValueError:
            hidden_layers = [16]
        num_inputs = 10
        num_outputs = len(Action)
        self.mlp_arch = [num_inputs] + hidden_layers + [num_outputs]

        if OPENCL_AVAILABLE:
            print("OpenCL available. Initializing MLPBatchProcessor.")
            self.batch_processor = MLPBatchProcessor(self.population_size, self.mlp_arch, verbose=True)
        else:
            print("OpenCL not available. Using CPU-only MLP.")
            self.batch_processor = None

    def _initialize_population(self):
        self.units = []
        start_x, start_y = 5, self.tile_map.grid_height // 2
        for i in range(self.population_size):
            brain = MLP(self.mlp_arch)
            self.units.append(SimplifiedUnit(i, start_x, start_y, brain))
            if self.batch_processor:
                self.batch_processor.update_brain_on_gpu(i, brain)
        self.tile_map.update_dynamic_grid(self.units, self.target)

    def _create_walls(self):
        # ... (same as before)
        if self.tile_map.grid_width > 20:
            for y in range(5, self.tile_map.grid_height - 5): self.tile_map.set_tile(20, y, Tile.WALL)
        if self.tile_map.grid_width > 25:
            for x in range(25, self.tile_map.grid_width - 5): self.tile_map.set_tile(x, 10, Tile.WALL)

    def update_settings(self, settings):
        new_mlp_arch_str = settings.get('mlp_arch_str', self.mlp_arch_str)
        new_pop_size = int(settings.get('population_size', self.population_size))

        arch_changed = new_mlp_arch_str != self.mlp_arch_str
        pop_changed = new_pop_size != self.population_size

        if arch_changed or (pop_changed and self.batch_processor):
            self.mlp_arch_str = new_mlp_arch_str
            self.population_size = new_pop_size
            self._setup_mlp_arch(self.mlp_arch_str)
            self._initialize_population()
            print("MLP architecture or population size changed. Population reset.")
        elif pop_changed:
            # Adjust population on CPU
            start_x, start_y = 5, self.tile_map.grid_height // 2
            if new_pop_size > self.population_size:
                for i in range(self.population_size, new_pop_size):
                    self.units.append(SimplifiedUnit(i, start_x, start_y, MLP(self.mlp_arch)))
            else:
                self.units = self.units[:new_pop_size]
            self.population_size = new_pop_size
            for i, unit in enumerate(self.units): unit.id = i
            print(f"Population size changed to {self.population_size}.")

        self.perception_radius = int(settings.get('perception_radius', self.perception_radius))
        self.steps_per_generation = int(settings.get('steps_per_gen', self.steps_per_generation))
        self.mutation_rate = float(settings.get('mutation_rate', self.mutation_rate))
        self.exploration_bonus = float(settings.get('exploration_bonus', self.exploration_bonus))

    def run_simulation_step(self):
        if not self.units: return

        if self.batch_processor:
            self._run_step_gpu()
        else:
            self._run_step_cpu()

    def _run_step_cpu(self):
        results = []
        for unit in self.units:
            inputs = self._get_unit_inputs(unit)
            action_probs, _ = unit.brain.forward(inputs)
            action = Action(np.argmax(action_probs))
            final_x, final_y = determine_new_position(unit.x, unit.y, action, self.tile_map.static_grid)
            results.append((unit.id, final_x, final_y))
        self.update_simulation_with_results(results)

    def _run_step_gpu(self):
        inputs_batch = np.array([self._get_unit_inputs(u) for u in self.units], dtype=np.float32)

        outputs_batch = self.batch_processor.forward_batch(inputs_batch)
        if outputs_batch is None:
            # Fallback if forward_batch fails
            return self._run_step_cpu()

        actions = np.argmax(outputs_batch, axis=1)

        results = []
        for i, unit in enumerate(self.units):
            action = Action(actions[i])
            final_x, final_y = determine_new_position(unit.x, unit.y, action, self.tile_map.static_grid)
            results.append((unit.id, final_x, final_y))
        self.update_simulation_with_results(results)

    def _get_unit_inputs(self, unit):
        vision = get_vision_inputs(unit.x, unit.y, self.tile_map.static_grid, self.perception_radius)
        dx = (self.target[0] - unit.x) / self.tile_map.grid_width
        dy = (self.target[1] - unit.y) / self.tile_map.grid_height
        return np.concatenate((vision, [dx, dy]))

    def update_simulation_with_results(self, results):
        for unit_id, new_x, new_y in results:
            if unit_id < len(self.units):
                self.units[unit_id].x = new_x
                self.units[unit_id].y = new_y
                self.units[unit_id].visited_tiles.add((new_x, new_y))
        self.tile_map.update_dynamic_grid(self.units, self.target)

    def evolve_population(self):
        fitness_scores = [self._calculate_fitness(u) for u in self.units]

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
        for i, unit in enumerate(self.units):
            unit.id = i
            if self.batch_processor:
                self.batch_processor.update_brain_on_gpu(i, unit.brain)

        self.generation += 1
        self.tile_map.update_dynamic_grid(self.units, self.target)
        print(f"Generation {self.generation} complete. Best fitness: {max(fitness_scores):.4f}")

    def _calculate_fitness(self, unit):
        dist_sq = (unit.x - self.target[0])**2 + (unit.y - self.target[1])**2
        proximity = 1 / (dist_sq + 1)
        exploration = self.exploration_bonus * len(unit.visited_tiles)
        return proximity + exploration

def get_vision_inputs(start_x, start_y, static_grid, vision_range):
    # ... (same as before)
    directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    vision_inputs = np.zeros(8)
    grid_width, grid_height = static_grid.shape
    for i, (dx, dy) in enumerate(directions):
        dist = 1
        found_wall = False
        while dist <= vision_range:
            check_x, check_y = start_x + dx * dist, start_y + dy * dist
            if not (0 <= check_x < grid_width and 0 <= check_y < grid_height):
                vision_inputs[i] = dist / vision_range
                found_wall = True
                break
            if static_grid[check_x, check_y] == Tile.WALL.value:
                vision_inputs[i] = dist / vision_range
                found_wall = True
                break
            dist += 1
        if not found_wall:
            vision_inputs[i] = 1.0
    return vision_inputs

def determine_new_position(unit_x, unit_y, action, static_grid):
    # ... (same as before)
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
