"""
This file contains the core logic for the simplified, tile-based simulation.
It is designed to be independent of Pygame for easier testing and multiprocessing.
"""
import numpy as np
from enum import Enum
from mlp import MLP
from mlp_batch_processor import MLPBatchProcessor, OPENCL_AVAILABLE

# --- Enums ---
class Tile(Enum): EMPTY = 0; WALL = 1; RESOURCE = 2
class Action(Enum): MOVE_N, MOVE_E, MOVE_S, MOVE_W, STAY, MINE = range(6)

# --- Simulation Classes ---
class TileMap:
    def __init__(self, grid_width, grid_height, static_grid=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        if static_grid is not None:
            self.static_grid = static_grid.copy()
            self.initial_static_grid = static_grid.copy()
        else:
            self.static_grid = np.full((self.grid_width, self.grid_height), Tile.EMPTY.value, dtype=int)
            self.initial_static_grid = self.static_grid.copy()

    def set_tile(self, x, y, tile_type: Tile):
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            self.static_grid[x, y] = tile_type.value
            self.initial_static_grid[x, y] = tile_type.value

    def is_wall(self, x, y):
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.static_grid[x, y] == Tile.WALL.value
        return True

    def is_resource(self, x, y):
        if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
            return self.static_grid[x, y] == Tile.RESOURCE.value
        return False

    def remove_resource(self, x, y):
        if self.is_resource(x, y):
            self.static_grid[x, y] = Tile.EMPTY.value
            return True
        return False

    def reset_map(self):
        self.static_grid = self.initial_static_grid.copy()

class SimplifiedUnit:
    def __init__(self, id, x, y, brain: MLP):
        self.id, self.x, self.y, self.brain = id, x, y, brain
        self.visited_tiles = set([(x, y)])
        self.last_action = Action.STAY
        self.mining_timer = 0
        self.inventory = 0
        self.visited_resource_tiles = set()

    def clone(self):
        cloned = SimplifiedUnit(self.id, self.x, self.y, self.brain.clone())
        cloned.last_action = self.last_action
        cloned.mining_timer = self.mining_timer
        cloned.inventory = self.inventory
        cloned.visited_resource_tiles = self.visited_resource_tiles.copy()
        cloned.returned_resources = 0
        return cloned

class SimplifiedGame:
    def __init__(self, width=40, height=30, population_size=100, mlp_arch_str="16",
                 perception_radius=5, steps_per_gen=100, mutation_rate=0.05,
                 proximity_bonus=1.0, exploration_bonus=0.0,
                 proximity_func='Inverse Squared', exploration_func='Linear',
                 static_grid=None, reward_system='Navigation'):
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
        self.bank = 0

        self.perception_radius = perception_radius
        self.steps_per_generation = steps_per_gen
        self.mutation_rate = mutation_rate
        self.proximity_bonus = proximity_bonus
        self.exploration_bonus = exploration_bonus
        self.proximity_func = proximity_func
        self.exploration_func = exploration_func
        self.mlp_arch_str = mlp_arch_str
        self.reward_system = reward_system

        self._setup_mlp_arch(mlp_arch_str)
        self._initialize_population()
        if static_grid is None: self._create_walls()

    def _setup_mlp_arch(self, mlp_arch_str):
        try:
            hidden_layers = [int(n.strip()) for n in mlp_arch_str.split(',') if n.strip()]
        except ValueError: hidden_layers = [16]
        # 8 for walls, 8 for resources, 2 for target vector, 1 for dist, 6 for last action, 1 for inventory
        num_inputs = 8 + 8 + 2 + 1 + len(Action) + 1
        self.mlp_arch = [num_inputs] + hidden_layers + [len(Action)]
        if OPENCL_AVAILABLE:
            if self.batch_processor and (self.batch_processor.pop_size != self.population_size or self.batch_processor.layers != self.mlp_arch):
                self.batch_processor.release()
                self.batch_processor = None
            if not self.batch_processor:
                self.batch_processor = MLPBatchProcessor(self.population_size, self.mlp_arch, verbose=True)
        else:
            self.batch_processor = None

    def _initialize_population(self):
        self.units = []
        self.tile_map.reset_map()
        for i in range(self.population_size):
            brain = MLP(self.mlp_arch)
            unit = SimplifiedUnit(i, self.spawn_point[0], self.spawn_point[1], brain)
            self.units.append(unit)
            if self.batch_processor:
                self.batch_processor.update_brain_on_gpu(i, brain)

    def _create_walls(self):
        if self.tile_map.grid_width > 20:
            for y in range(5, self.tile_map.grid_height - 5): self.tile_map.set_tile(20, y, Tile.WALL)
        if self.tile_map.grid_width > 25:
            for x in range(25, self.tile_map.grid_width - 5): self.tile_map.set_tile(x, 10, Tile.WALL)
        # Add some resource tiles
        for i in range(5):
            self.tile_map.set_tile(30 + i, 15, Tile.RESOURCE)
            self.tile_map.set_tile(10, 10 + i, Tile.RESOURCE)


    def restart(self):
        self.generation = 0
        self.fittest_brain = None
        self.best_fitness = 0.0
        self.average_fitness = 0.0
        self.best_fitness_components = (0.0, 0.0)
        self.fitness_history = []
        self.bank = 0
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
        self.reward_system = settings.get('reward_system', self.reward_system)

    def run_simulation_step(self):
        if not self.units: return
        (self._run_step_gpu if self.batch_processor else self._run_step_cpu)()

    def _run_step_cpu(self):
        results = []
        for unit in self.units:
            mined_resource = False
            if unit.mining_timer > 0:
                unit.mining_timer -= 1
                action = Action.MINE
                if unit.mining_timer == 0:
                    if self.tile_map.remove_resource(unit.x, unit.y):
                        mined_resource = True
            else:
                inputs = self._get_unit_inputs(unit)
                action_probs, _ = unit.brain.forward(inputs)
                action = Action(np.argmax(action_probs))
                if action == Action.MINE:
                    if self.tile_map.is_resource(unit.x, unit.y):
                        unit.mining_timer = 5
                    else: # penalize for trying to mine empty space
                        action = Action.STAY

            unit.last_action = action
            final_x, final_y = determine_new_position(unit.x, unit.y, action, self.tile_map)
            results.append((unit.id, final_x, final_y, mined_resource))
        self._update_units_from_results(results)

    def _run_step_gpu(self):
        inputs_batch = np.array([self._get_unit_inputs(u) for u in self.units if u.mining_timer == 0], dtype=np.float16)

        active_units = [u for u in self.units if u.mining_timer == 0]

        if not active_units:
             outputs_batch = []
        else:
             outputs_batch = self.batch_processor.forward_batch(inputs_batch)
             if outputs_batch is None: return self._run_step_cpu()

        actions = np.argmax(outputs_batch, axis=1)
        results = []

        active_unit_idx = 0
        for i, unit in enumerate(self.units):
            mined_resource = False
            action = unit.last_action

            if unit.mining_timer > 0:
                unit.mining_timer -= 1
                action = Action.MINE
                if unit.mining_timer == 0:
                    if self.tile_map.remove_resource(unit.x, unit.y):
                        mined_resource = True
            elif active_unit_idx < len(actions):
                action = Action(actions[active_unit_idx])
                if action == Action.MINE:
                    if self.tile_map.is_resource(unit.x, unit.y):
                        unit.mining_timer = 5
                    else:
                        action = Action.STAY
                active_unit_idx += 1

            unit.last_action = action
            final_x, final_y = determine_new_position(unit.x, unit.y, action, self.tile_map)
            results.append((unit.id, final_x, final_y, mined_resource))
        self._update_units_from_results(results)

    def _get_unit_inputs(self, unit):
        wall_vision = get_vision_inputs(unit.x, unit.y, self.tile_map, self.perception_radius, Tile.WALL)
        resource_vision = get_vision_inputs(unit.x, unit.y, self.tile_map, self.perception_radius, Tile.RESOURCE)

        dx_norm = (self.target[0] - unit.x) / self.tile_map.grid_width
        dy_norm = (self.target[1] - unit.y) / self.tile_map.grid_height
        target_vector = np.array([dx_norm, dy_norm])
        distance = np.sqrt(dx_norm**2 + dy_norm**2)

        last_action_vector = np.zeros(len(Action))
        last_action_vector[unit.last_action.value] = 1.0

        inventory_level = np.array([unit.inventory / 10.0]) # Assuming max inventory is 10 for normalization

        return np.concatenate((wall_vision, resource_vision, target_vector, last_action_vector, [distance], inventory_level))

    def _update_units_from_results(self, results):
        for unit_id, new_x, new_y, mined_resource in results:
            if unit_id < len(self.units):
                unit = self.units[unit_id]
                unit.x, unit.y = new_x, new_y
                unit.visited_tiles.add((new_x, new_y))
                if self.tile_map.is_resource(new_x, new_y):
                    unit.visited_resource_tiles.add((new_x, new_y))
                if mined_resource:
                    unit.inventory += 1

    def evolve_population(self):
        for unit in self.units:
            if (unit.x, unit.y) == self.spawn_point and unit.inventory > 0:
                self.bank += unit.inventory
                # This is a bit of a hack, but we'll use the proximity component to store the reward for returning resources
                unit.returned_resources = unit.inventory
                unit.inventory = 0
            else:
                unit.returned_resources = 0

        if self.reward_system == 'Navigation':
            all_fitness_data = [self._calculate_fitness(u) for u in self.units]
        else: # Resource Collection
            all_fitness_data = [self._calculate_resource_fitness(u) for u in self.units]

        fitness_scores = [f[0] for f in all_fitness_data]

        if not fitness_scores:
            self.best_fitness = 0
            self.average_fitness = 0
        else:
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

    def _calculate_resource_fitness(self, unit):
        # Reward for visiting tiles adjacent to resources
        resource_adj_tiles_visited = len(unit.visited_resource_tiles)
        exploration_score = resource_adj_tiles_visited * 0.1

        # Reward for returning resources to base
        return_bonus = unit.returned_resources * 10.0

        # A small reward for exploration
        general_exploration_score = len(unit.visited_tiles) * 0.01

        total_fitness = exploration_score + return_bonus + general_exploration_score
        return total_fitness, return_bonus, exploration_score

def get_vision_inputs(start_x, start_y, tile_map, vision_range, tile_type):
    directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    vision = np.zeros(8)
    for i, (dx, dy) in enumerate(directions):
        for dist in range(1, vision_range + 1):
            check_x, check_y = start_x + dx * dist, start_y + dy * dist
            is_obstacle = False
            if tile_type == Tile.WALL:
                is_obstacle = tile_map.is_wall(check_x, check_y)
            elif tile_type == Tile.RESOURCE:
                is_obstacle = tile_map.is_resource(check_x, check_y)

            if is_obstacle:
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
    elif action == Action.STAY or action == Action.MINE:
        pass # No change in position

    new_x, new_y = x + dx, y + dy
    return (new_x, new_y) if not tile_map.is_wall(new_x, new_y) else (x, y)
