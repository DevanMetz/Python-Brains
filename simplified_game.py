"""
This file contains the core logic for the simplified, tile-based simulation.
It is designed to be independent of Pygame for easier testing and multiprocessing.
"""
import numpy as np
from enum import Enum
from mlp import MLP
from mlp_batch_processor import MLPBatchProcessor, OPENCL_AVAILABLE

# --- Enums ---
class Tile(Enum): EMPTY = 0; WALL = 1
class Action(Enum): MOVE_N, MOVE_E, MOVE_S, MOVE_W, STAY = range(5)

# --- Simulation Classes ---
class TileMap:
    def __init__(self, grid_width, grid_height, static_grid=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        if static_grid is not None:
            self.static_grid = static_grid
        else:
            self.static_grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY.value, dtype=int)

    def set_tile(self, x, y, tile_type: Tile):
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self.static_grid[x, y] = tile_type.value

    def is_wall(self, x, y):
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.static_grid[x, y] == Tile.WALL.value
        return True

class SimplifiedUnit:
    def __init__(self, id, x, y, brain: MLP):
        self.id, self.x, self.y, self.brain = id, x, y, brain
        self.visited_tiles = set([(x, y)])
        self.last_action = Action.STAY

    def clone(self):
        cloned = SimplifiedUnit(self.id, self.x, self.y, self.brain.clone())
        cloned.last_action = self.last_action
        return cloned

class SimplifiedGame:
    def __init__(self, width=40, height=30, population_size=100, mlp_arch_str="16",
                 perception_radius=5, steps_per_gen=100, mutation_rate=0.05,
                 proximity_bonus=1.0, exploration_bonus=0.0,
                 proximity_func='Inverse Squared', exploration_func='Linear',
                 static_grid=None):
        self.tile_map = TileMap(width, height, static_grid)
        self.units = []
        self.target = (width - 5, height // 2)
        self.spawn_point = (5, height // 2)
        self.population_size = population_size
        self.generation = 0
        self.fittest_brain = None
        self.batch_processor = None
        self.best_fitness = 0.0
        self.average_fitness = 0.0
        self.best_fitness_components = (0.0, 0.0) # (proximity, exploration)
        self.fitness_history = []

        self.perception_radius = perception_radius
        self.steps_per_generation = steps_per_gen
        self.mutation_rate = mutation_rate
        self.proximity_bonus = proximity_bonus
        self.exploration_bonus = exploration_bonus
        self.proximity_func = proximity_func
        self.exploration_func = exploration_func
        self.mlp_arch_str = mlp_arch_str

        self._setup_mlp_arch(mlp_arch_str)
        self._initialize_population()
        if static_grid is None: self._create_walls()

    def _setup_mlp_arch(self, mlp_arch_str):
        try:
            hidden_layers = [int(n.strip()) for n in mlp_arch_str.split(',') if n.strip()]
        except ValueError: hidden_layers = [16]
        num_inputs = 15
        self.mlp_arch = [num_inputs] + hidden_layers + [len(Action)]
        if OPENCL_AVAILABLE:
            self.batch_processor = MLPBatchProcessor(self.population_size, self.mlp_arch, verbose=True)
        else:
            self.batch_processor = None

    def _initialize_population(self):
        self.units = []
        for i in range(self.population_size):
            brain = MLP(self.mlp_arch)
            self.units.append(SimplifiedUnit(i, self.spawn_point[0], self.spawn_point[1], brain))
            if self.batch_processor:
                self.batch_processor.update_brain_on_gpu(i, brain)

    def _create_walls(self):
        if self.tile_map.grid_width > 20:
            for y in range(5, self.tile_map.grid_height - 5): self.tile_map.set_tile(20, y, Tile.WALL)
        if self.tile_map.grid_width > 25:
            for x in range(25, self.tile_map.grid_width - 5): self.tile_map.set_tile(x, 10, Tile.WALL)

    def restart(self):
        self.generation = 0
        self.fittest_brain = None
        self.best_fitness = 0.0
        self.average_fitness = 0.0
        self.best_fitness_components = (0.0, 0.0)
        self.fitness_history = []
        self._initialize_population()
        print("Simulation restarted.")

    def update_settings(self, settings):
        new_mlp_arch_str = settings.get('mlp_arch_str', self.mlp_arch_str)
        new_pop_size = int(settings.get('population_size', self.population_size))
        arch_changed = new_mlp_arch_str != self.mlp_arch_str
        pop_changed = new_pop_size != self.population_size

        if arch_changed or (pop_changed and self.batch_processor):
            self.mlp_arch_str, self.population_size = new_mlp_arch_str, new_pop_size
            self._setup_mlp_arch(self.mlp_arch_str)
            self._initialize_population()
            print("MLP architecture or population size changed. Population reset.")
        elif pop_changed:
            if new_pop_size > self.population_size:
                for i in range(self.population_size, new_pop_size):
                    self.units.append(SimplifiedUnit(i, self.spawn_point[0], self.spawn_point[1], MLP(self.mlp_arch)))
            else:
                self.units = self.units[:new_pop_size]
            self.population_size = new_pop_size
            for i, unit in enumerate(self.units): unit.id = i

        self.perception_radius = int(settings.get('perception_radius', self.perception_radius))
        self.steps_per_generation = int(settings.get('steps_per_gen', self.steps_per_generation))
        self.mutation_rate = float(settings.get('mutation_rate', self.mutation_rate))
        self.proximity_bonus = float(settings.get('proximity_bonus', self.proximity_bonus))
        self.exploration_bonus = float(settings.get('exploration_bonus', self.exploration_bonus))
        self.proximity_func = settings.get('proximity_func', self.proximity_func)
        self.exploration_func = settings.get('exploration_func', self.exploration_func)

    def run_simulation_step(self):
        if not self.units: return
        (self._run_step_gpu if self.batch_processor else self._run_step_cpu)()

    def _run_step_cpu(self):
        results = []
        for unit in self.units:
            inputs = self._get_unit_inputs(unit)
            action_probs, _ = unit.brain.forward(inputs)
            action = Action(np.argmax(action_probs))
            unit.last_action = action
            final_x, final_y = determine_new_position(unit.x, unit.y, action, self.tile_map)
            results.append((unit.id, final_x, final_y))
        self._update_units_from_results(results)

    def _run_step_gpu(self):
        inputs_batch = np.array([self._get_unit_inputs(u) for u in self.units], dtype=np.float16)
        outputs_batch = self.batch_processor.forward_batch(inputs_batch)
        if outputs_batch is None: return self._run_step_cpu()
        actions = np.argmax(outputs_batch, axis=1)
        results = []
        for i, unit in enumerate(self.units):
            action = Action(actions[i])
            unit.last_action = action
            final_x, final_y = determine_new_position(unit.x, unit.y, action, self.tile_map)
            results.append((unit.id, final_x, final_y))
        self._update_units_from_results(results)

    def _get_unit_inputs(self, unit):
        vision = get_vision_inputs(unit.x, unit.y, self.tile_map, self.perception_radius)
        dx_norm = (self.target[0] - unit.x) / self.tile_map.grid_width
        dy_norm = (self.target[1] - unit.y) / self.tile_map.grid_height
        target_vector = np.array([dx_norm, dy_norm])
        distance = np.sqrt(dx_norm**2 + dy_norm**2)
        last_action_vector = np.zeros(4)
        if unit.last_action.value < 4: last_action_vector[unit.last_action.value] = 1.0
        return np.concatenate((vision, target_vector, last_action_vector, [distance]))

    def _update_units_from_results(self, results):
        for unit_id, new_x, new_y in results:
            if unit_id < len(self.units):
                self.units[unit_id].x, self.units[unit_id].y = new_x, new_y
                self.units[unit_id].visited_tiles.add((new_x, new_y))

    def evolve_population(self):
        all_fitness_data = [self._calculate_fitness(u) for u in self.units]
        fitness_scores = [f[0] for f in all_fitness_data]

        self.best_fitness = np.max(fitness_scores)
        self.average_fitness = np.mean(fitness_scores)
        self.fitness_history.append(self.best_fitness)

        sorted_indices = np.argsort(fitness_scores)[::-1]
        best_unit_index = sorted_indices[0]

        self.fittest_brain = self.units[best_unit_index].brain.clone()
        self.best_fitness_components = (all_fitness_data[best_unit_index][1], all_fitness_data[best_unit_index][2])

        sorted_units = [self.units[i] for i in sorted_indices]
        num_elites = self.population_size // 10

        next_gen_units = [u.clone() for u in sorted_units[:num_elites]]
        for unit in next_gen_units:
            unit.x, unit.y = self.spawn_point
            unit.visited_tiles = set([self.spawn_point])

        elite_pool = sorted_units[:num_elites] if num_elites > 0 else [sorted_units[0]]
        while len(next_gen_units) < self.population_size:
            parent = np.random.choice(elite_pool)
            child_brain = parent.brain.clone()
            child_brain.mutate(self.mutation_rate, 0.1)
            next_gen_units.append(SimplifiedUnit(len(next_gen_units), self.spawn_point[0], self.spawn_point[1], child_brain))

        self.units = next_gen_units
        for i, unit in enumerate(self.units):
            unit.id = i
            if self.batch_processor: self.batch_processor.update_brain_on_gpu(i, unit.brain)
        self.generation += 1

    def _calculate_fitness(self, unit):
        # Proximity component
        if self.proximity_func == 'None':
            proximity_score = 0.0
        else:
            dist_sq = (unit.x - self.target[0])**2 + (unit.y - self.target[1])**2
            if self.proximity_func == 'Inverse':
                proximity_score = 1.0 / (np.sqrt(dist_sq) + 1.0)
            elif self.proximity_func == 'Exponential':
                proximity_score = np.exp(-0.1 * np.sqrt(dist_sq))
            elif self.proximity_func == 'Logarithmic':
                proximity_score = 1.0 / (np.log(dist_sq + 1) + 1)
            else: # Inverse Squared (default)
                proximity_score = 1.0 / (dist_sq + 1.0)

        # Exploration component
        if self.exploration_func == 'None':
            exploration_score = 0.0
        else:
            visited_count = len(unit.visited_tiles)
            if self.exploration_func == 'Square Root':
                exploration_score = np.sqrt(visited_count)
            else: # Linear (default)
                exploration_score = float(visited_count)

        total_fitness = (self.proximity_bonus * proximity_score) + (self.exploration_bonus * exploration_score)
        return total_fitness, proximity_score, exploration_score

def get_vision_inputs(start_x, start_y, tile_map, vision_range=5):
    directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    vision = np.zeros(8)
    for i, (dx, dy) in enumerate(directions):
        for dist in range(1, vision_range + 1):
            if tile_map.is_wall(start_x + dx * dist, start_y + dy * dist):
                vision[i] = dist / vision_range
                break
        else:
            vision[i] = 1.0
    return vision

def determine_new_position(x, y, action, tile_map):
    dx, dy = 0, 0
    if action == Action.MOVE_N: dy = -1
    elif action == Action.MOVE_E: dx = 1
    elif action == Action.MOVE_S: dy = 1
    elif action == Action.MOVE_W: dx = -1
    new_x, new_y = x + dx, y + dy
    return (new_x, new_y) if not tile_map.is_wall(new_x, new_y) else (x, y)
