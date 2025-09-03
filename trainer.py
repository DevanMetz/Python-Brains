import numpy as np
import json
import os
import pygame
from functools import partial
from multiprocessing import Pool, cpu_count
from game import Unit, Target, Wall, Enemy
from mlp_cupy import MLPCupy as MLP
from map import Tile
from quadtree import QuadTree, Rectangle

# --- Worker Function for Multiprocessing ---

_tile_map_global = None

def init_worker(tile_map):
    """
    Initializer for each worker process.
    Makes the tile_map a global variable in the worker.
    """
    global _tile_map_global
    _tile_map_global = tile_map


def get_unit_inputs(unit_data, local_objects_data, target_pos_data):
    """
    Calculates MLP inputs. Uses a vectorized CuPy implementation for whisker-object
    intersections if available, otherwise falls back to an iterative approach.
    """
    unit_pos = pygame.Vector2(unit_data['position'])
    unit_angle = unit_data['angle']
    num_whiskers = unit_data['num_whiskers']
    whisker_length = unit_data['whisker_length']
    perceivable_types = unit_data['perceivable_types']
    unit_velocity = pygame.Vector2(unit_data['velocity'])

    whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, num_whiskers) if num_whiskers > 1 else np.array([0])
    whisker_inputs = np.zeros((num_whiskers, len(perceivable_types)))
    whisker_debug_info = []

    try:
        # Use CuPy for vectorized calculations if available
        import cupy as cp
        from math_utils import vectorized_line_circle_intersection
        xp = cp

        # --- Prepare whisker data on GPU ---
        abs_angles_gpu = xp.asarray(unit_angle + whisker_angles)
        start_points_gpu = xp.asarray(unit_pos.xy).reshape(1, 2)
        end_points_gpu = start_points_gpu + xp.stack([
            xp.cos(abs_angles_gpu) * whisker_length,
            xp.sin(abs_angles_gpu) * whisker_length
        ], axis=1)

        # --- Prepare object data on GPU ---
        if local_objects_data:
            centers_gpu = xp.asarray([d['position'] for d in local_objects_data])
            radii_gpu = xp.asarray([d['size'] for d in local_objects_data])
            obj_distances, obj_indices = vectorized_line_circle_intersection(
                start_points_gpu, end_points_gpu, centers_gpu, radii_gpu, xp)
        else:
            obj_distances = xp.full(num_whiskers, xp.inf)
            obj_indices = xp.full(num_whiskers, -1, dtype=xp.int32)

        # --- Wall detection (still iterative for now) ---
        wall_distances = np.full(num_whiskers, np.inf)
        if "wall" in perceivable_types:
            for i, angle in enumerate(whisker_angles):
                start_point = unit_pos
                end_point = start_point + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(unit_angle + angle))
                dx, dy = end_point.x - start_point.x, end_point.y - start_point.y
                steps = int(max(abs(dx), abs(dy)))
                if steps > 0:
                    x_inc, y_inc = dx / steps, dy / steps
                    for j in range(steps):
                        x = start_point.x + j * x_inc
                        y = start_point.y + j * y_inc
                        if _tile_map_global.get_tile_at_pixel(x, y) == Tile.WALL:
                            wall_distances[i] = unit_pos.distance_to(pygame.Vector2(x, y))
                            break

        # --- Combine results ---
        wall_distances_gpu = xp.asarray(wall_distances)
        obj_hit_is_closer = obj_distances < wall_distances_gpu

        final_distances = xp.asnumpy(xp.where(obj_hit_is_closer, obj_distances, wall_distances_gpu))
        final_indices = xp.asnumpy(xp.where(obj_hit_is_closer, obj_indices, -2)) # -2 for wall

        # --- Final processing on CPU ---
        object_types = [d['type'] for d in local_objects_data]
        for i in range(num_whiskers):
            dist = final_distances[i]
            idx = final_indices[i]
            detected_type = None
            if dist != np.inf:
                detected_type = "wall" if idx == -2 else object_types[idx]

            if detected_type and detected_type in perceivable_types:
                type_index = perceivable_types.index(detected_type)
                clamped_dist = max(0, min(dist, whisker_length))
                whisker_inputs[i, type_index] = 1.0 - (clamped_dist / whisker_length)

    except Exception:
        # --- Fallback to original iterative method ---
        from math_utils import iterative_line_circle_intersection
        for i, whisker_angle in enumerate(whisker_angles):
            abs_angle = unit_angle + whisker_angle
            start_point = unit_pos
            end_point = unit_pos + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(abs_angle))

            closest_dist = whisker_length
            detected_type = None

            if "wall" in perceivable_types:
                dx, dy = end_point.x - start_point.x, end_point.y - start_point.y
                steps = int(max(abs(dx), abs(dy)))
                if steps > 0:
                    x_inc, y_inc = dx / steps, dy / steps
                    for j in range(steps):
                        x, y = start_point.x + j * x_inc, start_point.y + j * y_inc
                        if _tile_map_global.get_tile_at_pixel(x,y) == Tile.WALL:
                            closest_dist = unit_pos.distance_to(pygame.Vector2(x,y))
                            detected_type = "wall"
                            break

            for obj_data in local_objects_data:
                dist = iterative_line_circle_intersection(start_point, end_point, pygame.Vector2(obj_data['position']), obj_data['size'])
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    detected_type = obj_data['type']

            if detected_type and detected_type in perceivable_types:
                type_index = perceivable_types.index(detected_type)
                clamped_dist = max(0, min(closest_dist, whisker_length))
                whisker_inputs[i, type_index] = 1.0 - (clamped_dist / whisker_length)

    # --- Target and other inputs (common to both paths) ---
    target_pos = pygame.Vector2(target_pos_data)
    relative_vec = target_pos - unit_pos
    relative_vec = relative_vec.rotate(-np.rad2deg(unit_angle))
    norm_dx = np.clip(relative_vec.x / _tile_map_global.pixel_width, -1, 1)
    norm_dy = np.clip(relative_vec.y / _tile_map_global.pixel_height, -1, 1)
    target_inputs = np.array([norm_dx, norm_dy])

    flat_whisker_inputs = whisker_inputs.flatten()
    other_inputs = np.array([unit_velocity.length() / 2.0, unit_angle / (2 * np.pi)])

    final_inputs = np.concatenate((flat_whisker_inputs, other_inputs, target_inputs))
    return final_inputs, whisker_debug_info


def run_single_unit_step(unit_data, local_objects_data, target_position_data):
    """
    The main worker function, refactored to accept local world data from the Quadtree.
    """
    inputs, whisker_debug_info = get_unit_inputs(
        unit_data,
        local_objects_data,
        target_position_data
    )

    brain = unit_data['brain']
    actions = brain.forward(inputs)

    return {
        "id": unit_data['id'],
        "actions": actions,
        "whisker_debug_info": whisker_debug_info
    }


def run_single_unit_step_wrapper(task_data):
    """Unpacks a dictionary and calls the main worker function."""
    return run_single_unit_step(
        task_data['unit_data'],
        task_data['local_objects_data'],
        task_data['target_position_data']
    )


class TrainingMode:
    NAVIGATE = 1
    COMBAT = 2

class TrainingSimulation:
    """
    Manages the genetic algorithm training process.
    """
    def __init__(self, population_size, world_size, tile_map, num_whiskers=7, perceivable_types=None, whisker_length=150):
        self.population_size = population_size
        self.num_to_draw = population_size # Default to drawing all units
        self.world_width, self.world_height = world_size
        self.tile_map = tile_map
        self.generation = 0
        self.num_whiskers = num_whiskers
        self.whisker_length = whisker_length
        self.perceivable_types = perceivable_types if perceivable_types is not None else ["wall", "enemy", "unit"]
        self.training_mode = TrainingMode.NAVIGATE

        # Initialize the multiprocessing pool
        # This creates a set of worker processes that will be reused across generations
        # The initializer passes the tile_map to each worker just once
        try:
            self.pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.tile_map,))
        except (OSError, ImportError):
            # Fallback for environments where multiprocessing is not fully supported (e.g. some web-based editors)
            print("Warning: Multiprocessing pool failed to initialize. Running in single-threaded mode.")
            self.pool = None

        # Define the MLP architecture
        num_inputs = self.num_whiskers * len(self.perceivable_types) + 2 + 2
        self.mlp_architecture = [num_inputs, 16, 2]

        # Create world objects
        self.target = Target(self.world_width - 50, self.world_height / 2)
        self.enemy = Enemy(self.world_width - 100, self.world_height / 2 + 100)
        self.world_objects = [self.target, self.enemy]
        self.projectiles = []

        # Create the initial population
        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}

        # Create the Quadtree for spatial partitioning
        qt_boundary = Rectangle(self.world_width / 2, self.world_height / 2, self.world_width, self.world_height)
        self.quadtree = QuadTree(qt_boundary, capacity=4)

    def _create_initial_population(self):
        population = []
        for i in range(self.population_size):
            brain = MLP(self.mlp_architecture)
            unit = Unit(
                id=i, x=50, y=self.world_height / 2, brain=brain,
                num_whiskers=self.num_whiskers,
                whisker_length=self.whisker_length,
                perceivable_types=self.perceivable_types,
                tile_map=self.tile_map
            )
            population.append(unit)
        return population

    def run_generation_step(self):
        """
        Runs a single step of the simulation, using the Quadtree for collision optimization.
        """
        # 1. Populate the Quadtree
        self.quadtree.clear()
        all_objects = self.world_objects + self.population
        for obj in all_objects:
            self.quadtree.insert(obj)

        # 2. Create tasks for each unit with localized data from the Quadtree
        target_pos_data = (self.target.position.x, self.target.position.y)
        tasks = []
        for unit in self.population:
            # Query for objects in a wide area around the unit
            query_radius = unit.whisker_length + unit.size
            query_area = Rectangle(unit.position.x, unit.position.y, query_radius * 2, query_radius * 2)
            local_objects = self.quadtree.query(query_area)

            # Serialize the local objects, excluding the unit itself
            local_objects_data = [obj.to_dict() for obj in local_objects if obj.id != unit.id]

            tasks.append({
                'unit_data': unit.to_dict(),
                'local_objects_data': local_objects_data,
                'target_position_data': target_pos_data
            })

        # 3. Run the simulation step in parallel
        if self.pool:
            results = self.pool.map(run_single_unit_step_wrapper, tasks)
        else:
            # Single-threaded fallback
            init_worker(self.tile_map)
            results = [run_single_unit_step_wrapper(task) for task in tasks]

        # 4. Apply results to the units
        for result in results:
            unit = self.population_map[result['id']]
            unit.update(result['actions'], self.projectiles)
            # Restore whisker visualization data from the worker
            unit.whisker_debug_info = [
                {
                    'start': pygame.Vector2(info['start']),
                    'end': pygame.Vector2(info['end']),
                    'full_end': pygame.Vector2(info['full_end']),
                    'type': info['type']
                } for info in result['whisker_debug_info']
            ]
            unit.position.x = np.clip(unit.position.x, 0, self.world_width)
            unit.position.y = np.clip(unit.position.y, 0, self.world_height)

        # 5. Update projectiles using the Quadtree
        for proj in self.projectiles[:]: # Iterate over a copy
            proj.update()
            if proj.lifespan <= 0:
                self.projectiles.remove(proj)
                continue

            # Query for nearby objects to check for collisions
            query_range = proj.get_bounding_box()
            nearby_objects = self.quadtree.query(query_range)

            for obj in nearby_objects:
                # Only check for collisions with enemies
                if isinstance(obj, Enemy):
                    if proj.position.distance_to(obj.position) < obj.size:
                        damage = 10
                        obj.health -= damage
                        proj.owner.damage_dealt += damage
                        self.projectiles.remove(proj)
                        if obj.health <= 0:
                            print("Enemy defeated!")
                            # For now, just reset its health
                            obj.health = 100
                        break # Projectile is gone, stop checking this projectile

    def get_drawable_units(self):
        """
        Returns a slice of the current population to be drawn on screen.
        This ensures that the units shown are always the ones being actively simulated.
        """
        return self.population[:self.num_to_draw]

    def evolve_population(self, elitism_frac=0.1, mutation_rate=0.05):
        """
        Evaluates fitness based on the current training mode and creates a new generation.
        """
        fitness_scores = []
        if self.training_mode == TrainingMode.NAVIGATE:
            for unit in self.population:
                distance = unit.position.distance_to(self.target.position)
                fitness = (self.world_width - distance) ** 2
                fitness_scores.append((unit, fitness))
        elif self.training_mode == TrainingMode.COMBAT:
            for unit in self.population:
                distance = unit.position.distance_to(self.enemy.position)
                fitness = unit.damage_dealt * 100 + (self.world_width - distance)
                fitness_scores.append((unit, fitness))

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        new_population = []
        num_elites = int(self.population_size * elitism_frac)
        elite_units = [item[0] for item in fitness_scores[:num_elites]]

        for elite_unit in elite_units:
            new_unit = Unit(
                id=len(new_population),
                x=50, y=self.world_height / 2, brain=elite_unit.brain.clone(),
                num_whiskers=self.num_whiskers, perceivable_types=self.perceivable_types,
                whisker_length=self.whisker_length, tile_map=self.tile_map
            )
            new_population.append(new_unit)

        while len(new_population) < self.population_size:
            parent1 = fitness_scores[np.random.randint(0, self.population_size // 2)][0]
            parent2 = fitness_scores[np.random.randint(0, self.population_size // 2)][0]
            child_brain = MLP.crossover(parent1.brain, parent2.brain)
            child_brain.mutate(mutation_rate=mutation_rate)
            new_unit = Unit(
                id=len(new_population),
                x=50, y=self.world_height / 2, brain=child_brain,
                num_whiskers=self.num_whiskers, perceivable_types=self.perceivable_types,
                whisker_length=self.whisker_length, tile_map=self.tile_map
            )
            new_population.append(new_unit)

        self.population = new_population
        self.population_map = {unit.id: unit for unit in self.population}
        self.generation += 1
        return fitness_scores[0][1]

    def cleanup(self):
        """
        Cleans up resources, specifically the multiprocessing pool.
        """
        if self.pool:
            self.pool.close()
            self.pool.join()
            print("Multiprocessing pool cleaned up.")

    def rebuild_with_new_architecture(self, new_arch, num_whiskers, perceivable_types, whisker_length):
        """
        Re-initializes the simulation with a new MLP architecture and I/O config.
        """
        print(f"Creating new population with arch: {new_arch}, {num_whiskers} whiskers, {whisker_length} length, sensing: {perceivable_types}")
        self.mlp_architecture = new_arch
        self.num_whiskers = num_whiskers
        self.perceivable_types = perceivable_types
        self.whisker_length = whisker_length
        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}
        self.projectiles = []
        self.enemy.health = 100
        for unit in self.population:
            unit.damage_dealt = 0
        self.generation = 0

    def set_population_size(self, new_size):
        """
        Updates the population size and resets the simulation.
        """
        print(f"Setting new population size to: {new_size}")
        self.population_size = new_size
        # The number of drawn units should not exceed the new population size
        if self.num_to_draw > self.population_size:
            self.num_to_draw = self.population_size

        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}
        self.projectiles = []
        self.enemy.health = 100
        self.generation = 0

    def save_fittest_brain(self, filepath_prefix="saved_brains/brain"):
        """
        Saves the architecture and weights of the fittest brain in the population.
        """
        if not self.population:
            print("Warning: Population is empty. Cannot save brain.")
            return

        fitness_scores = []
        if self.training_mode == TrainingMode.NAVIGATE:
            for unit in self.population:
                distance = unit.position.distance_to(self.target.position)
                fitness = (self.world_width - distance) ** 2
                fitness_scores.append((unit, fitness))
        elif self.training_mode == TrainingMode.COMBAT:
            for unit in self.population:
                distance = unit.position.distance_to(self.enemy.position)
                fitness = unit.damage_dealt * 100 + (self.world_width - distance)
                fitness_scores.append((unit, fitness))

        if not fitness_scores:
            print("Warning: Could not determine fittest brain.")
            return

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        fittest_unit = fitness_scores[0][0]

        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        arch_data = {
            "layer_sizes": fittest_unit.brain.layer_sizes,
            "num_whiskers": fittest_unit.num_whiskers,
            "whisker_length": fittest_unit.whisker_length,
            "perceivable_types": fittest_unit.perceivable_types
        }
        json_path = f"{filepath_prefix}_arch.json"
        with open(json_path, 'w') as f:
            json.dump(arch_data, f, indent=4)

        weights_path = f"{filepath_prefix}_weights.npz"
        np.savez(weights_path, *fittest_unit.brain.weights, *fittest_unit.brain.biases)
        print(f"Saved fittest brain to {json_path} and {weights_path}")

    def load_brain_from_file(self, filepath_prefix="saved_brains/brain"):
        """
        Loads a brain from files and rebuilds the population with it.
        """
        json_path = f"{filepath_prefix}_arch.json"
        weights_path = f"{filepath_prefix}_weights.npz"

        if not os.path.exists(json_path) or not os.path.exists(weights_path):
            print(f"Error: Brain files not found at {filepath_prefix}")
            return

        with open(json_path, 'r') as f:
            arch_data = json.load(f)

        layer_sizes = arch_data["layer_sizes"]
        num_whiskers = arch_data["num_whiskers"]
        perceivable_types = arch_data.get("perceivable_types", ["wall", "enemy", "unit"])
        whisker_length = arch_data.get("whisker_length", 150)

        loaded_brain = MLP(layer_sizes)
        with np.load(weights_path) as data:
            num_weight_matrices = len(loaded_brain.weights)
            for i in range(num_weight_matrices):
                loaded_brain.weights[i] = data[f'arr_{i}']
            for i in range(len(loaded_brain.biases)):
                loaded_brain.biases[i] = data[f'arr_{i + num_weight_matrices}']

        self.rebuild_with_new_architecture(layer_sizes, num_whiskers, perceivable_types, whisker_length)

        # Create a new population where each unit gets a clone of the loaded brain
        new_population = []
        for i in range(self.population_size):
            new_unit = Unit(
                id=i, x=50, y=self.world_height / 2, brain=loaded_brain.clone(),
                num_whiskers=self.num_whiskers,
                whisker_length=self.whisker_length,
                perceivable_types=self.perceivable_types,
                tile_map=self.tile_map
            )
            new_population.append(new_unit)
        self.population = new_population
        self.population_map = {unit.id: unit for unit in self.population}

        print(f"Loaded brain from {filepath_prefix} and rebuilt population.")
