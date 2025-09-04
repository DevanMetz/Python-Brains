"""
This file contains the core logic for the simplified, tile-based simulation.
It is designed to be independent of Pygame for easier testing and multiprocessing.
"""
import numpy as np
import heapq
from enum import Enum
from mlp import MLP
from mlp_batch_processor import MLPBatchProcessor, OPENCL_AVAILABLE

# --- Enums ---
class Tile(Enum): EMPTY, WALL, RESOURCE, DROPOFF, ENEMY = range(5)
class Action(Enum): MOVE_TO_RESOURCE, GATHER, MOVE_TO_DROPOFF, FLEE, IDLE = range(5)

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
    def __init__(self, id, x, y, brain: MLP, max_health=10, max_cargo=10):
        self.id, self.x, self.y, self.brain = id, x, y, brain
        self.visited_tiles = set([(x, y)])

        # Resource collection state
        self.max_health = max_health
        self.health = self.max_health
        self.max_cargo = max_cargo
        self.cargo = 0
        self.is_carrying = False # Boolean flag for efficiency

        # Pathfinding and action state
        self.current_path = []
        self.last_action = Action.IDLE # This will be the high-level action
        self.goal_type = None
        self.goal_pos = None

        # Fitness tracking
        self.fitness_score = 0.0
        self.is_dead = False

    def clone(self):
        # When a unit is cloned for the next generation, it gets a fresh brain
        # and its state is reset by the __init__ method.
        cloned = SimplifiedUnit(self.id, self.x, self.y, self.brain.clone(), self.max_health, self.max_cargo)
        return cloned

class Enemy:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SimplifiedGame:
    def __init__(self, width=40, height=30, population_size=100, mlp_arch_str="16",
                 perception_radius=5, steps_per_gen=100, mutation_rate=0.05,
                 proximity_bonus=1.0, exploration_bonus=0.0,
                 proximity_func='Inverse Squared', exploration_func='Linear',
                 attack_awareness_radius=2, static_grid=None):
        self.tile_map = TileMap(width, height, static_grid)
        self.units = []
        self.enemies = []
        self.resource_locations = []
        self.dropoff_locations = []
        self.target = (width - 5, height // 2) # Note: Will be deprecated
        self.spawn_point = (5, height // 2)
        self.population_size = population_size
        self.generation = 0
        self.fittest_brain = None
        self.batch_processor = None
        self.best_fitness = 0.0
        self.average_fitness = 0.0
        self.best_fitness_components = (0.0, 0.0)
        self.fitness_history = []

        self.perception_radius = perception_radius
        self.steps_per_generation = steps_per_gen
        self.mutation_rate = mutation_rate
        self.proximity_bonus = proximity_bonus
        self.exploration_bonus = exploration_bonus
        self.proximity_func = proximity_func
        self.exploration_func = exploration_func
        self.attack_awareness_radius = attack_awareness_radius
        self.mlp_arch_str = mlp_arch_str

        self._setup_mlp_arch(mlp_arch_str)
        self._initialize_map_objects()
        self._initialize_population()
        if static_grid is None: self._create_walls()

        # Store a pristine copy of the grid for resets
        self.pristine_static_grid = np.copy(self.tile_map.static_grid)


    def _initialize_map_objects(self):
        self.enemies.clear()
        self.resource_locations.clear()
        self.dropoff_locations.clear()

        # Use the pristine grid to find spawn locations
        grid = self.pristine_static_grid if hasattr(self, 'pristine_static_grid') else self.tile_map.static_grid

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                tile = grid[x, y]
                if tile == Tile.ENEMY.value:
                    self.enemies.append(Enemy(x, y))
                elif tile == Tile.RESOURCE.value:
                    self.resource_locations.append((x,y))
                elif tile == Tile.DROPOFF.value:
                    self.dropoff_locations.append((x,y))

    def _setup_mlp_arch(self, mlp_arch_str):
        try:
            hidden_layers = [int(n.strip()) for n in mlp_arch_str.split(',') if n.strip()]
        except ValueError: hidden_layers = [16]
        num_inputs = 10 # Updated for the new input vector
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
        self._initialize_map_objects()
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
        self.attack_awareness_radius = int(settings.get('attack_awareness_radius', self.attack_awareness_radius))

    def run_simulation_step(self):
        if not self.units: return

        # Update enemies first, so units can react in the same step
        self._update_enemies()

        # --- Step 1: Get Action from MLP (GPU or CPU) ---
        # Create a list of active units to process
        active_units = [u for u in self.units if not u.is_dead]
        if not active_units: return

        if self.batch_processor:
            inputs_batch = np.array([self._get_unit_inputs(u) for u in active_units], dtype=np.float16)
            outputs_batch = self.batch_processor.forward_batch(inputs_batch)
            if outputs_batch is None: # Fallback to CPU
                actions = [Action(np.argmax(u.brain.forward(self._get_unit_inputs(u))[0])) for u in active_units]
            else:
                actions = [Action(a) for a in np.argmax(outputs_batch, axis=1)]
        else: # CPU path
            actions = [Action(np.argmax(u.brain.forward(self._get_unit_inputs(u))[0])) for u in active_units]

        for i, unit in enumerate(active_units):
            unit.last_action = actions[i]

        # --- Step 2: Execute High-Level Actions & Update State ---
        for unit in active_units:
            action = unit.last_action
            unit_pos = (unit.x, unit.y)

            # Movement actions
            if action == Action.MOVE_TO_RESOURCE:
                if not unit.is_carrying: unit.fitness_score += 1 # Reward for correct intention
                target_pos, _ = self._find_nearest_object(unit_pos, self.resource_locations)
                if target_pos and (unit.goal_type != 'resource' or unit.goal_pos != target_pos):
                    unit.goal_type, unit.goal_pos = 'resource', target_pos
                    unit.current_path = find_path(self.tile_map, unit_pos, target_pos)
                    if unit.current_path: unit.current_path.pop(0) # Remove current position
            elif action == Action.MOVE_TO_DROPOFF:
                if unit.is_carrying: unit.fitness_score += 1 # Reward for correct intention
                target_pos, _ = self._find_nearest_object(unit_pos, self.dropoff_locations)
                if target_pos and (unit.goal_type != 'dropoff' or unit.goal_pos != target_pos):
                    unit.goal_type, unit.goal_pos = 'dropoff', target_pos
                    unit.current_path = find_path(self.tile_map, unit_pos, target_pos)
                    if unit.current_path: unit.current_path.pop(0) # Remove current position
            elif action == Action.FLEE:
                unit.goal_type, unit.goal_pos = 'flee', None
                enemy_pos, _ = self._find_nearest_object(unit_pos, self.enemies)
                if enemy_pos:
                    escape_dx, escape_dy = unit_pos[0] - enemy_pos[0], unit_pos[1] - enemy_pos[1]
                    # Find a valid escape tile by checking along the escape vector
                    for i in range(5, 0, -1):
                        tx = unit.x + int(np.sign(escape_dx) * i)
                        ty = unit.y + int(np.sign(escape_dy) * i)
                        tx = max(0, min(self.tile_map.grid_width - 1, tx))
                        ty = max(0, min(self.tile_map.grid_height - 1, ty))
                        if not self.tile_map.is_wall(tx, ty):
                            unit.current_path = find_path(self.tile_map, unit_pos, (tx, ty))
                            if unit.current_path: unit.current_path.pop(0) # Remove current position
                            break # Found a valid target

            # Non-movement actions
            elif action == Action.GATHER:
                unit.current_path = []
                _, dist = self._find_nearest_object(unit_pos, self.resource_locations)
                if dist <= 1 and unit.cargo < unit.max_cargo:
                    unit.cargo += 1
                    unit.is_carrying = True
                    unit.fitness_score += 5 # Reward for successful gather
                else:
                    unit.fitness_score -= 10 # Penalty for invalid gather action
            elif action == Action.IDLE:
                unit.current_path = []
                if unit.cargo < unit.max_cargo:
                    unit.fitness_score -= 1 # Penalty for being idle

        # --- Step 3: Update Unit Positions from Paths ---
        results = []
        for unit in active_units:
            final_x, final_y = unit.x, unit.y
            if unit.current_path:
                # Move to the next step and consume it from the path
                final_x, final_y = unit.current_path.pop(0)

            results.append((unit.id, final_x, final_y))

        # Add non-moving dead units to results
        dead_units = [u for u in self.units if u.is_dead]
        for unit in dead_units:
            results.append((unit.id, unit.x, unit.y))

        self._update_units_from_results(results)

        # --- Step 4: Handle Interactions, Rewards, and Death ---
        self._handle_unit_interactions()


    def _handle_unit_interactions(self):
        for unit in self.units:
            # Handle death
            if unit.health <= 0 and not unit.is_dead:
                unit.is_dead = True
                unit.fitness_score -= 200 # Large penalty for dying
                continue # No other interactions for dead units

            # Handle resource drop-off
            if unit.cargo > 0:
                _, dist_to_dropoff = self._find_nearest_object((unit.x, unit.y), self.dropoff_locations)
                if dist_to_dropoff <= 1: # If at a dropoff point
                    if unit.cargo == unit.max_cargo:
                        unit.fitness_score += 100 # Large reward for full delivery
                    # Reset cargo after any delivery
                    unit.cargo = 0
                    unit.is_carrying = False


    def _update_enemies(self):
        living_units = [u for u in self.units if not u.is_dead]
        if not living_units:
            return # No one to hunt

        for enemy in self.enemies:
            # Find the closest unit from the living ones
            closest_unit = min(living_units, key=lambda u: (u.x - enemy.x)**2 + (u.y - enemy.y)**2)

            # Move towards the closest unit
            path = find_path(self.tile_map, (enemy.x, enemy.y), (closest_unit.x, closest_unit.y))
            if len(path) > 1:
                enemy.x, enemy.y = path[1] # The first element is the start position

            # Attack any adjacent units
            for unit in living_units: # Only check against living units
                if abs(unit.x - enemy.x) <= 1 and abs(unit.y - enemy.y) <= 1:
                    unit.health -= 1
                    unit.fitness_score -= 2 # Penalty for taking damage

    def _find_nearest_object(self, unit_pos, object_list):
        if not object_list:
            return None, float('inf')

        closest_obj = None
        min_dist_sq = float('inf')

        for obj in object_list:
            obj_pos = (obj.x, obj.y) if hasattr(obj, 'x') else obj
            dist_sq = (unit_pos[0] - obj_pos[0])**2 + (unit_pos[1] - obj_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_obj = obj_pos

        return closest_obj, np.sqrt(min_dist_sq)

    def _get_unit_inputs(self, unit):
        # 1. Unit's Own State
        health_norm = unit.health / unit.max_health
        cargo_norm = unit.cargo / unit.max_cargo if unit.max_cargo > 0 else 0.0
        is_carrying = 1.0 if unit.cargo > 0 else 0.0

        # 2. Task-Related State
        nearest_resource_pos, _ = self._find_nearest_object((unit.x, unit.y), self.resource_locations)
        if nearest_resource_pos:
            res_dx = (nearest_resource_pos[0] - unit.x) / self.tile_map.grid_width
            res_dy = (nearest_resource_pos[1] - unit.y) / self.tile_map.grid_height
        else: # No resources left
            res_dx, res_dy = 0.0, 0.0

        nearest_dropoff_pos, _ = self._find_nearest_object((unit.x, unit.y), self.dropoff_locations)
        if nearest_dropoff_pos:
            base_dx = (nearest_dropoff_pos[0] - unit.x) / self.tile_map.grid_width
            base_dy = (nearest_dropoff_pos[1] - unit.y) / self.tile_map.grid_height
        else: # Should not happen, but defensively code
            base_dx, base_dy = 0.0, 0.0

        # 3. Threat Awareness
        is_under_attack = 0.0
        for enemy in self.enemies:
            if abs(unit.x - enemy.x) <= self.attack_awareness_radius and abs(unit.y - enemy.y) <= self.attack_awareness_radius:
                is_under_attack = 1.0
                break

        nearest_enemy_pos, _ = self._find_nearest_object((unit.x, unit.y), self.enemies)
        if nearest_enemy_pos:
            enemy_dx = (nearest_enemy_pos[0] - unit.x) / self.tile_map.grid_width
            enemy_dy = (nearest_enemy_pos[1] - unit.y) / self.tile_map.grid_height
        else: # No enemies
            enemy_dx, enemy_dy = 0.0, 0.0

        return np.array([
            health_norm, is_carrying, cargo_norm,
            res_dx, res_dy, base_dx, base_dy,
            is_under_attack, enemy_dx, enemy_dy
        ])

    def _update_units_from_results(self, results):
        for unit_id, new_x, new_y in results:
            if unit_id < len(self.units):
                self.units[unit_id].x, self.units[unit_id].y = new_x, new_y
                self.units[unit_id].visited_tiles.add((new_x, new_y))

    def evolve_population(self):
        # Reset enemies and other map objects for the new generation
        self._initialize_map_objects()

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
        # The fitness is now the accumulated score from the reward system.
        # The component scores are returned as 0 since they are no longer used.
        return unit.fitness_score, 0.0, 0.0

def determine_new_position(x, y, action, tile_map):
    dx, dy = 0, 0
    if action == Action.MOVE_N: dy = -1
    elif action == Action.MOVE_E: dx = 1
    elif action == Action.MOVE_S: dy = 1
    elif action == Action.MOVE_W: dx = -1
    new_x, new_y = x + dx, y + dy
    return (new_x, new_y) if not tile_map.is_wall(new_x, new_y) else (x, y)

def find_path(tile_map, start, end):
    """
    Finds the shortest path between two points using the A* algorithm.
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start)) # (f_score, position)

    came_from = {}
    g_score = { (x,y): float('inf') for x in range(tile_map.grid_width) for y in range(tile_map.grid_height) }
    g_score[start] = 0

    f_score = { (x,y): float('inf') for x in range(tile_map.grid_width) for y in range(tile_map.grid_height) }
    f_score[start] = heuristic(start, end)

    open_set_hash = {start}

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < tile_map.grid_width and 0 <= neighbor[1] < tile_map.grid_height):
                continue

            if tile_map.is_wall(neighbor[0], neighbor[1]):
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    return [] # Return empty list if no path is found
