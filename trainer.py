"""
Manages the core training simulation, genetic algorithm, and multiprocessing.

This module contains the `TrainingSimulation` class, which orchestrates the
entire AI training process. It handles the creation of unit populations,
manages the simulation steps, evaluates fitness, and evolves the population
using a genetic algorithm.

It also defines the worker functions that are distributed to a multiprocessing
pool, allowing the simulation to run in parallel for significant performance
gains. This includes a sophisticated, vectorized approach to calculating unit
sensory inputs using CuPy for GPU acceleration where available.
"""
import numpy as np
import json
import os
import pygame
from functools import partial
from multiprocessing import Pool, cpu_count
from game import Unit, Target, Wall, Enemy
from mlp_opencl import MLPOpenCL as MLP, OPENCL_AVAILABLE
from map import Tile
from quadtree import QuadTree, Rectangle
from math_utils import iterative_line_circle_intersection

# Import OpenCL functions if available
if OPENCL_AVAILABLE:
    import pyopencl as cl
    from math_utils_opencl import opencl_unified_perception
    from mlp_opencl import context, queue


# --- Worker Function for Multiprocessing ---

_tile_map_global = None
_gpu_brain_cache = {}
_tile_map_buf_global = None # Cache for the tilemap OpenCL buffer

def init_worker(tile_map):
    """Initializer for each worker process in the multiprocessing pool.

    This function sets up global variables within the worker's scope.
    This is an optimization to avoid serializing and sending large, static
    data like the tile map with every task.

    Args:
        tile_map (TileMap): The global tile map for the simulation.
    """
    global _tile_map_global, _gpu_brain_cache
    _tile_map_global = tile_map
    # The brain cache is also made global to persist across tasks within
    # the same worker process. It stores GPU-resident copies of brain weights.
    _gpu_brain_cache = {}


def get_unit_inputs(unit_data, local_objects_data, target_pos_data):
    """
    Calculates all sensory inputs for a single unit's MLP brain.
    This function is now only used as a fallback for CPU-only mode.
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

    # --- Iterative method for CPU fallback ---
    for i, whisker_angle in enumerate(whisker_angles):
        abs_angle = unit_angle + whisker_angle
        start_point = unit_pos
        end_point = unit_pos + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(abs_angle))

        closest_dist = whisker_length
        detected_type = None
        intersect_point = end_point

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

        if closest_dist < whisker_length:
             intersect_point = start_point + pygame.Vector2(closest_dist, 0).rotate(np.rad2deg(abs_angle))

        if detected_type and detected_type in perceivable_types:
            type_index = perceivable_types.index(detected_type)
            clamped_dist = max(0, min(closest_dist, whisker_length))
            whisker_inputs[i, type_index] = 1.0 - (clamped_dist / whisker_length)

        whisker_debug_info.append({
            'start': (start_point.x, start_point.y),
            'end': (intersect_point.x, intersect_point.y),
            'full_end': (end_point.x, end_point.y),
            'type': detected_type
        })

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


def _process_perception_results(unit_data, whisker_results, circle_object_types):
    """
    Helper to process raw whisker results for a single unit.
    This converts distances and indices into the whisker_input vector and debug info.
    """
    unit_pos = pygame.Vector2(unit_data['position'])
    unit_angle = unit_data['angle']
    num_whiskers = unit_data['num_whiskers']
    whisker_length = unit_data['whisker_length']
    perceivable_types = unit_data['perceivable_types']

    distances, indices = whisker_results
    whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, num_whiskers) if num_whiskers > 1 else np.array([0])
    whisker_inputs = np.zeros((num_whiskers, len(perceivable_types)))
    whisker_debug_info = []

    for i in range(num_whiskers):
        dist, idx = distances[i], indices[i]
        detected_type = None
        abs_angle_rad = unit_angle + whisker_angles[i]
        start_point = unit_pos
        full_end_point = start_point + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(abs_angle_rad))
        intersect_point = full_end_point

        if dist != np.inf and dist <= whisker_length:
            if idx == -2:
                detected_type = "wall"
            # Explicitly check that the index is valid before using it
            elif idx != -1 and idx < len(circle_object_types):
                detected_type = circle_object_types[idx]

            # If a valid object was detected, process it
            if detected_type:
                intersect_point = start_point + pygame.Vector2(dist, 0).rotate(np.rad2deg(abs_angle_rad))
                if detected_type in perceivable_types:
                type_index = perceivable_types.index(detected_type)
                clamped_dist = max(0, min(dist, whisker_length))
                whisker_inputs[i, type_index] = 1.0 - (clamped_dist / whisker_length)

        whisker_debug_info.append({'start': (start_point.x, start_point.y), 'end': (intersect_point.x, intersect_point.y), 'full_end': (full_end_point.x, full_end_point.y), 'type': detected_type})

    return whisker_inputs, whisker_debug_info


def run_single_unit_step(unit_data, brain_id, mlp_inputs, whisker_debug_info):
    """
    The main worker function for processing one unit for one simulation step.
    This version receives pre-computed MLP inputs.
    """
    brain = unit_data['brain']
    actions = None
    if OPENCL_AVAILABLE:
        try:
            from mlp_opencl import context
            if brain_id not in _gpu_brain_cache:
                weights_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w.astype(np.float32)) for w in brain.weights]
                biases_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.astype(np.float32)) for b in brain.biases]
                max_layer_size = max(brain.layer_sizes)
                intermediate_buf_size = max_layer_size * np.dtype(np.float32).itemsize
                intermediate_buf_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=intermediate_buf_size)
                intermediate_buf_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=intermediate_buf_size)
                _gpu_brain_cache[brain_id] = {
                    'weights': weights_bufs, 'biases': biases_bufs,
                    'intermediate_a': intermediate_buf_a, 'intermediate_b': intermediate_buf_b
                }
            cached_buffers = _gpu_brain_cache[brain_id]
            actions = brain.forward(mlp_inputs, cached_buffers=cached_buffers)
        except Exception as e:
            print(f"OpenCL forward pass with caching failed: {e}. Falling back.")
            actions = brain.forward(mlp_inputs)
    else:
        actions = brain.forward(mlp_inputs)

    return { "id": unit_data['id'], "actions": actions, "whisker_debug_info": whisker_debug_info }


def clear_cache_worker(_):
    """A worker task to clear all global GPU caches."""
    global _gpu_brain_cache
    if _gpu_brain_cache:
        for brain_data in _gpu_brain_cache.values():
            for buf in brain_data['weights']: buf.release()
            for buf in brain_data['biases']: buf.release()
            if 'intermediate_a' in brain_data: brain_data['intermediate_a'].release()
            if 'intermediate_b' in brain_data: brain_data['intermediate_b'].release()
        _gpu_brain_cache.clear()
    return True


def run_single_unit_step_wrapper(task_data):
    """Unpacks a task dictionary and calls the main worker function."""
    return run_single_unit_step(
        task_data['unit_data'],
        task_data['brain_id'],
        task_data['mlp_inputs'],
        task_data['whisker_debug_info']
    )


class TrainingMode:
    """An enumeration for the different training modes available."""
    NAVIGATE = 1
    COMBAT = 2

class TrainingSimulation:
    """
    Manages the genetic algorithm training process for a population of units.
    """
    def __init__(self, population_size, world_size, tile_map, num_whiskers=7, perceivable_types=None, whisker_length=150):
        self.population_size = population_size
        self.num_to_draw = population_size
        self.world_width, self.world_height = world_size
        self.tile_map = tile_map
        self.generation = 0
        self.num_whiskers = num_whiskers
        self.whisker_length = whisker_length
        self.perceivable_types = perceivable_types if perceivable_types is not None else ["wall", "enemy", "unit"]
        self.training_mode = TrainingMode.NAVIGATE

        if not OPENCL_AVAILABLE:
            print("\n" + "="*60 + "\n WARNING: GPU ACCELERATION NOT AVAILABLE \n" + "="*60)
            # ... (warning message)

        try:
            self.pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.tile_map,))
        except (OSError, ImportError):
            self.pool = None

        num_inputs = self.num_whiskers * len(self.perceivable_types) + 2 + 2
        self.mlp_architecture = [num_inputs, 16, 2]
        self.target = Target(self.world_width - 50, self.world_height / 2)
        self.enemy = Enemy(self.world_width - 100, self.world_height / 2 + 100)
        self.world_objects = [self.target, self.enemy]
        self.projectiles = []
        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}
        qt_boundary = Rectangle(self.world_width / 2, self.world_height / 2, self.world_width, self.world_height)
        self.quadtree = QuadTree(qt_boundary, capacity=4)

        self.perception_cache = {}
        self.tile_map_buf = None
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

        # Initialize globals for the main process as well (for the CPU path)
        init_worker(self.tile_map)

    def _init_gpu_buffers(self):
        """Initializes all persistent OpenCL buffers for the simulation."""
        print("Initializing persistent OpenCL buffers...")
        if self.tile_map_buf: self.tile_map_buf.release()
        get_tile_value = np.vectorize(lambda t: t.value)
        int_grid = get_tile_value(self.tile_map.grid)
        map_data = int_grid.T.flatten().astype(np.int32)
        self.tile_map_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=map_data)

        if self.perception_cache:
            for buf in self.perception_cache.values(): buf.release()
        total_whiskers = self.population_size * self.num_whiskers
        max_circles = self.population_size + len(self.world_objects)
        mf = cl.mem_flags
        self.perception_cache['p1s_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, total_whiskers * 8))
        self.perception_cache['p2s_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, total_whiskers * 8))
        self.perception_cache['centers_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, max_circles * 8))
        self.perception_cache['radii_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, max_circles * 4))
        self.perception_cache['out_distances_buf'] = cl.Buffer(context, mf.WRITE_ONLY, size=max(1, total_whiskers * 4))
        self.perception_cache['out_indices_buf'] = cl.Buffer(context, mf.WRITE_ONLY, size=max(1, total_whiskers * 4))
        self.perception_cache['whisker_parent_indices_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, total_whiskers * 4))
        print("SUCCESS: Initialized OpenCL perception cache for batched processing.")

    def _create_initial_population(self):
        population = []
        for i in range(self.population_size):
            brain = MLP(self.mlp_architecture)
            unit = Unit(id=i, x=50, y=self.world_height / 2, brain=brain, num_whiskers=self.num_whiskers, whisker_length=self.whisker_length, perceivable_types=self.perceivable_types, tile_map=self.tile_map)
            population.append(unit)
        return population

    def run_generation_step(self):
        """Runs a single frame/step of the simulation for the entire population."""
        self.quadtree.clear()
        all_objects = self.world_objects + self.population
        for obj in all_objects: self.quadtree.insert(obj)

        tasks = []
        target_pos_data = (self.target.position.x, self.target.position.y)

        # --- Vectorized Perception Path (OpenCL) ---
        if OPENCL_AVAILABLE and self.pool:
            total_whiskers = self.population_size * self.num_whiskers
            all_p1s = np.empty((total_whiskers, 2), dtype=np.float32)
            all_p2s = np.empty((total_whiskers, 2), dtype=np.float32)
            whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_whiskers) if self.num_whiskers > 1 else np.array([0])

            for i, unit in enumerate(self.population):
                start_idx, end_idx = i * self.num_whiskers, (i + 1) * self.num_whiskers
                abs_angles = unit.angle + whisker_angles
                all_p1s[start_idx:end_idx] = np.array([unit.position.x, unit.position.y])
                end_vectors = np.column_stack([np.cos(abs_angles) * unit.whisker_length, np.sin(abs_angles) * unit.whisker_length])
                all_p2s[start_idx:end_idx] = all_p1s[start_idx:end_idx] + end_vectors

            circle_objects = [obj for obj in all_objects if isinstance(obj, (Unit, Enemy, Target))]
            # Create a mapping from a unit's ID to its index in the circle_objects list
            circle_indices_map = {obj.id: i for i, obj in enumerate(circle_objects) if isinstance(obj, Unit)}

            all_whisker_parent_indices = np.empty(total_whiskers, dtype=np.int32)
            for i, unit in enumerate(self.population):
                start_idx, end_idx = i * self.num_whiskers, (i + 1) * self.num_whiskers
                parent_circle_idx = circle_indices_map.get(unit.id, -1) # -1 if not found
                all_whisker_parent_indices[start_idx:end_idx] = parent_circle_idx

            if circle_objects:
                centers_np = np.array([o.position for o in circle_objects], dtype=np.float32)
                radii_np = np.array([o.size for o in circle_objects], dtype=np.float32)
            else:
                centers_np = np.empty((0, 2), dtype=np.float32)
                radii_np = np.empty(0, dtype=np.float32)

            cl.enqueue_copy(queue, self.perception_cache['p1s_buf'], all_p1s, is_blocking=False)
            cl.enqueue_copy(queue, self.perception_cache['p2s_buf'], all_p2s, is_blocking=False)
            cl.enqueue_copy(queue, self.perception_cache['centers_buf'], centers_np, is_blocking=False)
            cl.enqueue_copy(queue, self.perception_cache['radii_buf'], radii_np, is_blocking=False)
            cl.enqueue_copy(queue, self.perception_cache['whisker_parent_indices_buf'], all_whisker_parent_indices, is_blocking=False)

            detect_walls = "wall" in self.perceivable_types
            detect_circles = any(t in self.perceivable_types for t in ["enemy", "target", "unit"])

            kernel_event = opencl_unified_perception(
                queue, self.perception_cache['p1s_buf'], self.perception_cache['p2s_buf'], self.perception_cache['centers_buf'], self.perception_cache['radii_buf'],
                self.perception_cache['out_distances_buf'], self.perception_cache['out_indices_buf'],
                self.perception_cache['whisker_parent_indices_buf'],
                total_whiskers, len(circle_objects), self.tile_map_buf, self.tile_map.grid_width, self.tile_map.tile_size,
                detect_circles, detect_walls
            )

            all_distances = np.empty(total_whiskers, dtype=np.float32)
            all_indices = np.empty(total_whiskers, dtype=np.int32)
            # Read results back from the GPU. This must wait for the kernel to finish.
            cl.enqueue_copy(queue, all_distances, self.perception_cache['out_distances_buf'], wait_for=[kernel_event])
            cl.enqueue_copy(queue, all_indices, self.perception_cache['out_indices_buf'], wait_for=[kernel_event]).wait()

            circle_object_types = [o.type for o in circle_objects]
            for i, unit in enumerate(self.population):
                start_idx, end_idx = i * self.num_whiskers, (i + 1) * self.num_whiskers
                whisker_results = (all_distances[start_idx:end_idx], all_indices[start_idx:end_idx])

                unit_data = unit.to_dict()
                whisker_inputs, whisker_debug_info = _process_perception_results(unit_data, whisker_results, circle_object_types)

                relative_vec = self.target.position - unit.position
                relative_vec = relative_vec.rotate(-np.rad2deg(unit.angle))
                target_inputs = np.array([np.clip(relative_vec.x / self.tile_map.pixel_width, -1, 1), np.clip(relative_vec.y / self.tile_map.pixel_height, -1, 1)])
                other_inputs = np.array([unit.velocity.length() / unit.speed, unit.angle / (2 * np.pi)])
                mlp_inputs = np.concatenate((whisker_inputs.flatten(), other_inputs, target_inputs))

                tasks.append({'unit_data': unit_data, 'brain_id': id(unit.brain), 'mlp_inputs': mlp_inputs, 'whisker_debug_info': whisker_debug_info})

        # --- Fallback Path (CPU) ---
        else:
            for unit in self.population:
                query_radius = unit.whisker_length + unit.size
                query_area = Rectangle(unit.position.x, unit.position.y, query_radius * 2, query_radius * 2)
                local_objects = self.quadtree.query(query_area)
                local_objects_data = [obj.to_dict() for obj in local_objects if obj.id != unit.id]

                unit_data = unit.to_dict()
                mlp_inputs, whisker_debug_info = get_unit_inputs(unit_data, local_objects_data, target_pos_data)
                tasks.append({'unit_data': unit_data, 'brain_id': id(unit.brain), 'mlp_inputs': mlp_inputs, 'whisker_debug_info': whisker_debug_info})

        # --- Run simulation step in parallel ---
        if self.pool:
            results = self.pool.map(run_single_unit_step_wrapper, tasks)
        else:
            init_worker(self.tile_map)
            results = [run_single_unit_step_wrapper(task) for task in tasks]

        # --- Apply results ---
        for result in results:
            unit = self.population_map[result['id']]
            unit.update(result['actions'], self.projectiles)
            unit.whisker_debug_info = [{'start': pygame.Vector2(info['start']), 'end': pygame.Vector2(info['end']), 'full_end': pygame.Vector2(info['full_end']), 'type': info['type']} for info in result['whisker_debug_info']]
            unit.position.x = np.clip(unit.position.x, 0, self.world_width)
            unit.position.y = np.clip(unit.position.y, 0, self.world_height)

        # --- Update projectiles ---
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
        """Returns a slice of the population to be drawn on screen.

        This is used to improve rendering performance by only drawing a subset
        of the total population, while still simulating all of them.

        Returns:
            list[Unit]: The sub-list of units to be rendered.
        """
        return self.population[:self.num_to_draw]

    def evolve_population(self, elitism_frac=0.1, mutation_rate=0.05):
        """Evaluates fitness and creates a new generation of units.

        This method implements the genetic algorithm:
        1.  Calculates a fitness score for each unit based on the current
            `training_mode`.
        2.  Sorts the population by fitness.
        3.  Selects the top-performing units ("elites") to pass directly to
            the next generation, unchanged.
        4.  Fills the rest of the new population by repeatedly selecting two
            high-fitness parents, performing crossover on their brains to
            create a child, and mutating the child's brain.
        5.  Replaces the old population with the new one.

        Args:
            elitism_frac (float, optional): The fraction of the population to
                carry over as elites. Defaults to 0.1.
            mutation_rate (float, optional): The probability for each weight
                or bias in a child's brain to be mutated. Defaults to 0.05.

        Returns:
            float: The fitness score of the best-performing unit in the
                   previous generation.
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

        # Clear the GPU cache in the worker processes, as the old brains are now gone.
        self.clear_worker_caches()

        return fitness_scores[0][1]

    def clear_worker_caches(self):
        """Sends a task to each worker to clear its GPU cache."""
        if self.pool and OPENCL_AVAILABLE:
            # The argument to map is an iterable of inputs for the function.
            # We just need to run the function once for each worker.
            num_workers = self.pool._processes
            self.pool.map(clear_cache_worker, range(num_workers))
            # print("Cleared GPU caches in worker processes.") # Optional: for debugging

    def cleanup(self):
        """
        Cleans up resources, specifically terminating the multiprocessing pool
        and releasing any OpenCL buffers.
        This should be called before the application exits.
        """
        if self.pool:
            self.pool.close()
            self.pool.join()
            print("Multiprocessing pool cleaned up.")
            self.pool = None

        if OPENCL_AVAILABLE:
            # Release perception buffers
            if self.perception_cache:
                for buf in self.perception_cache.values():
                    buf.release()
                self.perception_cache.clear()
            # Release map buffer
            if self.tile_map_buf:
                self.tile_map_buf.release()
                self.tile_map_buf = None
            print("Released OpenCL buffers.")

    def rebuild_pool(self):
        """
        Shuts down and recreates the multiprocessing pool.

        This is necessary after the tile map has been edited to ensure that
        the worker processes are re-initialized with the updated map data.
        """
        self.cleanup()
        try:
            self.pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.tile_map,))
            print("Multiprocessing pool rebuilt successfully.")
        except (OSError, ImportError):
            print("Warning: Multiprocessing pool failed to initialize. Running in single-threaded mode.")
            self.pool = None

        # Also rebuild the master GPU buffers with the new map data
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

    def rebuild_with_new_architecture(self, new_arch, num_whiskers, perceivable_types, whisker_length):
        """Resets the entire simulation with a new unit brain configuration.

        This method is called when the user designs a new AI in the UI. It
        replaces the current population with a brand new, randomly initialized
        one based on the specified architecture.

        Args:
            new_arch (list[int]): The MLP layer sizes for the new brains.
            num_whiskers (int): The number of whiskers for the new units.
            perceivable_types (list[str]): The object types the new units can see.
            whisker_length (int): The whisker length for the new units.
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

        # Clear the GPU cache in the worker processes, as the old brains are now gone.
        self.clear_worker_caches()

        # Re-initialize GPU buffers with new sizes
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

    def set_population_size(self, new_size):
        """Updates the population size and resets the simulation.

        Args:
            new_size (int): The desired total number of units.
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

        # Re-initialize GPU buffers with new sizes
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

    def save_fittest_brain(self, filepath_prefix="saved_brains/brain"):
        """Saves the architecture and weights of the current fittest brain.

        Calculates fitness for the current population, identifies the best
        performing unit, and saves its brain's architecture (`.json`) and
        weights/biases (`.npz`) to disk.

        Args:
            filepath_prefix (str, optional): The base path and filename for the
                saved files. Defaults to "saved_brains/brain".
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
        """Loads a brain from files and replaces the current population with it.

        Reads a saved brain's architecture (`.json`) and weights (`.npz`),
        then rebuilds the entire population. Each unit in the new population
        receives a clone of the loaded brain.

        Args:
            filepath_prefix (str, optional): The base path and filename for the
                brain files to load. Defaults to "saved_brains/brain".
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
