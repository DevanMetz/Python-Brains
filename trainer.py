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


def _rasterize_objects_on_grid(grid, objects, tile_map):
    """Draws objects onto a grid for grid-based perception."""
    from map import ObjectType
    for obj in objects:
        # All objects are now 1x1 tiles
        if 0 <= obj.grid_x < tile_map.grid_width and 0 <= obj.grid_y < tile_map.grid_height:
            type_map = { "unit": ObjectType.UNIT, "enemy": ObjectType.ENEMY, "target": ObjectType.TARGET }
            obj_type_val = type_map.get(obj.type)
            if obj_type_val:
                # Don't overwrite walls
                if grid[obj.grid_x, obj.grid_y] == 0:
                    grid[obj.grid_x, obj.grid_y] = obj_type_val.value

def _process_perception_results(unit_data, whisker_results, whisker_length):
    """
    Helper to process raw whisker results for a single unit.
    This version uses object types returned directly from the kernel and
    formats the data for the new grid-based MLP input layer.
    """
    from map import ObjectType
    num_whiskers = 8

    # Each whisker provides 2 inputs: normalized distance and object type
    whisker_inputs = np.zeros(num_whiskers * 2, dtype=np.float32)
    whisker_debug_info = []

    distances, object_types_from_kernel = whisker_results

    # Fixed 8 directions for whiskers
    whisker_directions = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]

    for i in range(num_whiskers):
        dist, obj_type_val = distances[i], object_types_from_kernel[i]

        # Normalize distance: 1.0 means no object, 0.0 means object is right next to unit
        normalized_dist = dist / whisker_length

        whisker_inputs[i*2] = normalized_dist
        whisker_inputs[i*2 + 1] = float(obj_type_val)

        # --- Update debug info ---
        start_pos = (unit_data['grid_x'], unit_data['grid_y'])
        direction = whisker_directions[i]
        full_end_pos = (start_pos[0] + direction[0] * whisker_length, start_pos[1] + direction[1] * whisker_length)

        detected_type = None
        end_pos = full_end_pos
        if dist != np.inf:
            type_map = {
                ObjectType.WALL.value: "wall", ObjectType.UNIT.value: "unit",
                ObjectType.ENEMY.value: "enemy", ObjectType.TARGET.value: "target"
            }
            detected_type = type_map.get(obj_type_val)
            end_pos = (start_pos[0] + direction[0] * int(dist), start_pos[1] + direction[1] * int(dist))

        whisker_debug_info.append({'start': start_pos, 'end': end_pos, 'full_end': full_end_pos, 'type': detected_type})

    return whisker_inputs, whisker_debug_info


def run_single_unit_step(unit_data, brain_id, mlp_inputs, whisker_debug_info):
    """The main worker function. Receives MLP inputs and runs the forward pass."""
    brain = unit_data['brain']
    actions = None
    if OPENCL_AVAILABLE:
        try:
            # The MLP OpenCL caching logic remains the same
            from mlp_opencl import context
            if brain_id not in _gpu_brain_cache:
                weights_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w.astype(np.float32)) for w in brain.weights]
                biases_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.astype(np.float32)) for b in brain.biases]
                max_layer_size = max(brain.layer_sizes)
                intermediate_buf_size = max_layer_size * np.dtype(np.float32).itemsize
                intermediate_buf_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=intermediate_buf_size)
                intermediate_buf_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=intermediate_buf_size)
                _gpu_brain_cache[brain_id] = {'weights': weights_bufs, 'biases': biases_bufs, 'intermediate_a': intermediate_buf_a, 'intermediate_b': intermediate_buf_b}
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
    def __init__(self, population_size, world_size, tile_map):
        self.population_size = population_size
        self.num_to_draw = population_size
        self.world_width, self.world_height = world_size
        self.tile_map = tile_map
        self.generation = 0
        self.training_mode = TrainingMode.NAVIGATE

        # --- New Simplified, Fixed Architecture ---
        self.num_whiskers = 8
        self.whisker_length = 10 # in tiles
        # Each whisker provides 2 inputs: distance and object type
        num_inputs = self.num_whiskers * 2 + 2 # +2 for target vector
        # 8 outputs for 8 directions of movement
        num_outputs = 8
        self.mlp_architecture = [num_inputs, 16, num_outputs]

        if not OPENCL_AVAILABLE:
            print("\n" + "="*60 + "\n WARNING: GPU ACCELERATION NOT AVAILABLE \n" + "="*60)
            # ... (warning message)

        try:
            self.pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.tile_map,))
        except (OSError, ImportError):
            self.pool = None

        # Create world objects at grid positions
        target_x = self.tile_map.grid_width - 5
        target_y = self.tile_map.grid_height // 2
        self.target = Target(target_x, target_y, self.tile_map)

        enemy_x = self.tile_map.grid_width - 10
        enemy_y = self.tile_map.grid_height // 2 + 5
        self.enemy = Enemy(enemy_x, enemy_y, self.tile_map)

        self.world_objects = [self.target, self.enemy]
        self.projectiles = [] # This will be unused but kept for now to avoid breaking other code.
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
        if self.perception_cache:
            for buf in self.perception_cache.values(): buf.release()

        total_whiskers = self.population_size * self.num_whiskers
        mf = cl.mem_flags

        # The tile_map_buf is now the world_grid_buf and must be writeable
        grid_size_bytes = self.tile_map.grid.nbytes
        self.tile_map_buf = cl.Buffer(context, mf.READ_WRITE, size=max(1, grid_size_bytes))

        # Simplified perception cache
        self.perception_cache['p1s_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, total_whiskers * 8))
        self.perception_cache['p2s_buf'] = cl.Buffer(context, mf.READ_WRITE, size=max(1, total_whiskers * 8))
        self.perception_cache['out_distances_buf'] = cl.Buffer(context, mf.WRITE_ONLY, size=max(1, total_whiskers * 4))
        self.perception_cache['out_indices_buf'] = cl.Buffer(context, mf.WRITE_ONLY, size=max(1, total_whiskers * 4))
        print("SUCCESS: Initialized OpenCL perception cache for batched processing.")

    def _create_initial_population(self):
        population = []
        start_x = 5
        start_y = self.tile_map.grid_height // 2
        for i in range(self.population_size):
            brain = MLP(self.mlp_architecture)
            unit = Unit(id=i, grid_x=start_x, grid_y=start_y, brain=brain, tile_map=self.tile_map, whisker_length=self.whisker_length)
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
            # 1. Prepare whisker data
            total_whiskers = self.population_size * self.num_whiskers
            all_p1s = np.empty((total_whiskers, 2), dtype=np.float32)
            all_p2s = np.empty((total_whiskers, 2), dtype=np.float32)
            whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_whiskers) if self.num_whiskers > 1 else np.array([0])

            for i, unit in enumerate(self.population):
                start_idx, end_idx = i * self.num_whiskers, (i + 1) * self.num_whiskers
                abs_angles = unit.angle + whisker_angles
                start_offset_vectors = np.column_stack([np.cos(abs_angles) * unit.size, np.sin(abs_angles) * unit.size])
                p1_positions = np.array([unit.position.x, unit.position.y]) + start_offset_vectors
                all_p1s[start_idx:end_idx] = p1_positions
                end_vectors = np.column_stack([np.cos(abs_angles) * unit.whisker_length, np.sin(abs_angles) * unit.whisker_length])
                all_p2s[start_idx:end_idx] = p1_positions + end_vectors

            # 2. Prepare world grid by rasterizing objects onto it
            # Start with a copy of the base wall map
            dynamic_grid_int = np.array([t.value for t in self.tile_map.grid.flat], dtype=np.int32).reshape(self.tile_map.grid.shape)
            circles_to_draw = [obj for obj in all_objects if 'size' in obj.__dict__ and obj.type != 'unit']
            # Rasterize enemies and targets first
            _rasterize_circles_on_grid(dynamic_grid_int, circles_to_draw, self.tile_map.tile_size)
            # Rasterize units on top, so they take precedence
            _rasterize_circles_on_grid(dynamic_grid_int, self.population, self.tile_map.tile_size)

            # 3. Upload data to GPU
            cl.enqueue_copy(queue, self.perception_cache['p1s_buf'], all_p1s, is_blocking=False)
            cl.enqueue_copy(queue, self.perception_cache['p2s_buf'], all_p2s, is_blocking=False)
            cl.enqueue_copy(queue, self.tile_map_buf, dynamic_grid_int.T.flatten(), is_blocking=False) # Upload dynamic grid

            # 4. Execute Kernel
            kernel_event = opencl_unified_perception(
                queue, self.perception_cache['p1s_buf'], self.perception_cache['p2s_buf'],
                self.tile_map_buf, self.perception_cache['out_distances_buf'], self.perception_cache['out_indices_buf'],
                total_whiskers, self.tile_map.grid_width, self.tile_map.tile_size
            )

            # 5. Read results back
            all_distances = np.empty(total_whiskers, dtype=np.float32)
            all_indices = np.empty(total_whiskers, dtype=np.int32)
            cl.enqueue_copy(queue, all_distances, self.perception_cache['out_distances_buf'], wait_for=[kernel_event])
            cl.enqueue_copy(queue, all_indices, self.perception_cache['out_indices_buf'], wait_for=[kernel_event]).wait()

            # 6. Process results for each unit
            for i, unit in enumerate(self.population):
                start_idx, end_idx = i * self.num_whiskers, (i + 1) * self.num_whiskers
                whisker_results = (all_distances[start_idx:end_idx], all_indices[start_idx:end_idx])

                unit_data = unit.to_dict()
                whisker_inputs, whisker_debug_info = _process_perception_results(unit_data, whisker_results)

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
                # Use Manhattan distance on the grid
                distance = abs(unit.grid_x - self.target.grid_x) + abs(unit.grid_y - self.target.grid_y)
                max_dist = self.tile_map.grid_width + self.tile_map.grid_height
                fitness = (max_dist - distance) ** 2
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

    def reset_simulation(self):
        """Resets the simulation to its initial state."""
        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}
        self.generation = 0
        # Clear the GPU cache in the worker processes, as the old brains are now gone.
        self.clear_worker_caches()

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
        # Architecture is now fixed, so we only need to save weights.
        weights_path = f"{filepath_prefix}_weights.npz"
        np.savez(weights_path, *fittest_unit.brain.weights, *fittest_unit.brain.biases)
        print(f"Saved fittest brain to {weights_path}")

    def load_brain_from_file(self, filepath_prefix="saved_brains/brain"):
        """Loads a brain's weights and replaces the current population with it."""
        weights_path = f"{filepath_prefix}_weights.npz"

        if not os.path.exists(weights_path):
            print(f"Error: Brain weights file not found at {weights_path}")
            return

        # Create a new brain with the fixed architecture
        loaded_brain = MLP(self.mlp_architecture)
        with np.load(weights_path) as data:
            num_weight_matrices = len(loaded_brain.weights)
            for i in range(num_weight_matrices):
                loaded_brain.weights[i] = data[f'arr_{i}']
            for i in range(len(loaded_brain.biases)):
                loaded_brain.biases[i] = data[f'arr_{i + num_weight_matrices}']

        # Create a new population where each unit gets a clone of the loaded brain
        new_population = []
        start_x = 5
        start_y = self.tile_map.grid_height // 2
        for i in range(self.population_size):
            unit = Unit(id=i, grid_x=start_x, grid_y=start_y, brain=loaded_brain.clone(), tile_map=self.tile_map, whisker_length=self.whisker_length)
            new_population.append(unit)

        self.population = new_population
        self.population_map = {unit.id: unit for unit in self.population}
        self.generation = 0
        self.clear_worker_caches()
        print(f"Loaded brain from {weights_path} and rebuilt population.")
