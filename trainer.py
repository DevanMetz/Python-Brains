"""
Manages the core training simulation, genetic algorithm, and multiprocessing.

This module contains the `TrainingSimulation` class, which orchestrates the
entire AI training process. It handles the creation of unit populations,
manages the simulation steps, evaluates fitness, and evolves the population
using a genetic algorithm.
"""
import numpy as np
import json
import os
import pygame
from functools import partial
from multiprocessing import Pool, cpu_count
from game import Unit, Target, Wall, Enemy
from mlp_opencl import MLPOpenCL as MLP, OPENCL_AVAILABLE
from map import Tile, ObjectType

# Import OpenCL functions if available
if OPENCL_AVAILABLE:
    import pyopencl as cl
    from math_utils_opencl import opencl_unified_perception
    from mlp_opencl import context, queue


# --- Worker Function for Multiprocessing ---

_tile_map_global = None
_gpu_brain_cache = {}

def init_worker(tile_map):
    """Initializer for each worker process in the multiprocessing pool."""
    global _tile_map_global, _gpu_brain_cache
    _tile_map_global = tile_map
    _gpu_brain_cache = {}

def _rasterize_objects_on_grid(grid, objects, tile_map):
    """Draws objects onto a grid for grid-based perception."""
    for obj in objects:
        if 0 <= obj.grid_x < tile_map.grid_width and 0 <= obj.grid_y < tile_map.grid_height:
            type_map = { "unit": ObjectType.UNIT, "enemy": ObjectType.ENEMY, "target": ObjectType.TARGET }
            obj_type_val = type_map.get(obj.type)
            if obj_type_val:
                if grid[obj.grid_x, obj.grid_y] == 0:
                    grid[obj.grid_x, obj.grid_y] = obj_type_val.value

def _process_perception_results(unit_data, whisker_results, whisker_length):
    """
    Helper to process raw whisker results for a single unit.
    Formats the data for the new grid-based MLP input layer.
    """
    num_whiskers = 8
    whisker_inputs = np.zeros(num_whiskers * 2, dtype=np.float32)
    whisker_debug_info = []
    distances, object_types_from_kernel = whisker_results

    whisker_directions = [
        (0, -1), (1, -1), (1, 0), (1, 1),
        (0, 1), (-1, 1), (-1, 0), (-1, -1)
    ]

    for i in range(num_whiskers):
        dist, obj_type_val = distances[i], object_types_from_kernel[i]
        normalized_dist = dist / whisker_length if whisker_length > 0 else 0.0
        whisker_inputs[i*2] = normalized_dist
        whisker_inputs[i*2 + 1] = float(obj_type_val)

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
        task_data['unit_data'], task_data['brain_id'],
        task_data['mlp_inputs'], task_data['whisker_debug_info']
    )

class TrainingMode:
    NAVIGATE = 1
    COMBAT = 2

class TrainingSimulation:
    def __init__(self, population_size, world_size, tile_map):
        self.population_size = population_size
        self.num_to_draw = population_size
        self.world_width, self.world_height = world_size
        self.tile_map = tile_map
        self.generation = 0
        self.training_mode = TrainingMode.NAVIGATE
        self.num_whiskers = 8
        self.whisker_length = 10
        num_inputs = self.num_whiskers * 2 + 2
        num_outputs = 8
        self.mlp_architecture = [num_inputs, 16, num_outputs]

        if not OPENCL_AVAILABLE:
            print("\n" + "="*60 + "\n WARNING: GPU ACCELERATION NOT AVAILABLE \n" + "="*60)

        try:
            self.pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.tile_map,))
        except (OSError, ImportError):
            self.pool = None

        target_x = self.tile_map.grid_width - 5
        target_y = self.tile_map.grid_height // 2
        self.target = Target(target_x, target_y, self.tile_map)
        enemy_x = self.tile_map.grid_width - 10
        enemy_y = self.tile_map.grid_height // 2 + 5
        self.enemy = Enemy(enemy_x, enemy_y, self.tile_map)

        self.world_objects = [self.target, self.enemy]
        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}

        self.perception_cache = {}
        self.tile_map_buf = None
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

        init_worker(self.tile_map)

    def _init_gpu_buffers(self):
        print("Initializing persistent OpenCL buffers...")
        if self.tile_map_buf: self.tile_map_buf.release()
        if self.perception_cache:
            for buf in self.perception_cache.values(): buf.release()

        total_whiskers = self.population_size * self.num_whiskers
        mf = cl.mem_flags
        grid_size_bytes = self.tile_map.grid.nbytes
        self.tile_map_buf = cl.Buffer(context, mf.READ_WRITE, size=max(1, grid_size_bytes))
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
        if not (OPENCL_AVAILABLE and self.pool):
            print("WARNING: OpenCL or multiprocessing pool not available. Skipping generation step.")
            return

        all_dynamic_objects = self.population + self.world_objects
        tasks = []

        # 1. Rasterize world state onto a grid
        # We start with a fresh copy of the static grid (walls)
        dynamic_grid_int = self.tile_map.grid.copy().astype(np.int32)
        _rasterize_objects_on_grid(dynamic_grid_int, all_dynamic_objects, self.tile_map)

        # 2. Prepare whisker data for all units in a batch
        total_whiskers = self.population_size * self.num_whiskers
        all_p1s = np.empty((total_whiskers, 2), dtype=np.float32)
        all_p2s = np.empty((total_whiskers, 2), dtype=np.float32)

        for i, unit in enumerate(self.population):
            start_idx = i * self.num_whiskers
            # All 8 whiskers start at the unit's grid center (in pixel coordinates)
            p1_pixel = (float(unit.grid_x * self.tile_map.tile_size + self.tile_map.tile_size/2),
                        float(unit.grid_y * self.tile_map.tile_size + self.tile_map.tile_size/2))
            all_p1s[start_idx : start_idx + self.num_whiskers] = p1_pixel

            for j, direction in enumerate(unit.whisker_directions):
                # Whisker endpoints are calculated in pixel coordinates
                p2_pixel = (p1_pixel[0] + direction[0] * unit.whisker_length * self.tile_map.tile_size,
                            p1_pixel[1] + direction[1] * unit.whisker_length * self.tile_map.tile_size)
                all_p2s[start_idx + j] = p2_pixel

        # 3. Upload data to GPU
        # The .T.flatten() is because OpenCL expects a flat array in column-major order,
        # while numpy defaults to row-major. Transposing aligns them.
        cl.enqueue_copy(queue, self.tile_map_buf, dynamic_grid_int.T.flatten(), is_blocking=False)
        cl.enqueue_copy(queue, self.perception_cache['p1s_buf'], all_p1s, is_blocking=False)
        cl.enqueue_copy(queue, self.perception_cache['p2s_buf'], all_p2s, is_blocking=False)

        # 4. Execute Kernel
        kernel_event = opencl_unified_perception(
            queue, self.perception_cache['p1s_buf'], self.perception_cache['p2s_buf'],
            self.tile_map_buf, self.perception_cache['out_distances_buf'], self.perception_cache['out_indices_buf'],
            total_whiskers, self.tile_map.grid_width, self.tile_map.tile_size
        )

        # 5. Read results back
        all_distances = np.empty(total_whiskers, dtype=np.float32)
        all_object_types = np.empty(total_whiskers, dtype=np.int32)
        cl.enqueue_copy(queue, all_distances, self.perception_cache['out_distances_buf'], wait_for=[kernel_event])
        cl.enqueue_copy(queue, all_object_types, self.perception_cache['out_indices_buf'], wait_for=[kernel_event]).wait()

        # 6. Process results and create tasks for the multiprocessing pool
        for i, unit in enumerate(self.population):
            start_idx = i * self.num_whiskers
            end_idx = start_idx + self.num_whiskers
            whisker_results = (all_distances[start_idx:end_idx], all_object_types[start_idx:end_idx])

            unit_data = unit.to_dict()
            whisker_inputs, whisker_debug_info = _process_perception_results(unit_data, whisker_results, unit.whisker_length)

            # Add target vector to the MLP inputs
            target_vec = np.array([self.target.grid_x - unit.grid_x, self.target.grid_y - unit.grid_y], dtype=np.float32)
            norm_factor = max(self.tile_map.grid_width, self.tile_map.grid_height)
            if norm_factor > 0:
                target_vec /= norm_factor

            mlp_inputs = np.concatenate((whisker_inputs, target_vec))
            tasks.append({
                'unit_data': unit_data,
                'brain_id': id(unit.brain),
                'mlp_inputs': mlp_inputs,
                'whisker_debug_info': whisker_debug_info
            })

        # 7. Run MLP forward pass for all units in parallel
        if tasks:
            results = self.pool.map(run_single_unit_step_wrapper, tasks)
        else:
            results = []

        # 8. Apply results to the units
        for result in results:
            unit = self.population_map[result['id']]
            unit.update(result['actions'])
            unit.whisker_debug_info = result['whisker_debug_info']
            # Ensure units stay within bounds
            unit.grid_x = np.clip(unit.grid_x, 0, self.tile_map.grid_width - 1)
            unit.grid_y = np.clip(unit.grid_y, 0, self.tile_map.grid_height - 1)

    def get_drawable_units(self):
        return self.population[:self.num_to_draw]

    def evolve_population(self, elitism_frac=0.1, mutation_rate=0.05):
        fitness_scores = []
        # Simplified fitness for grid-based navigation
        for unit in self.population:
            distance = abs(unit.grid_x - self.target.grid_x) + abs(unit.grid_y - self.target.grid_y)
            max_dist = self.tile_map.grid_width + self.tile_map.grid_height
            fitness = (max_dist - distance) ** 2
            fitness_scores.append((unit, fitness))

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        new_population = []
        num_elites = int(self.population_size * elitism_frac)
        elite_units = [item[0] for item in fitness_scores[:num_elites]]

        for elite_unit in elite_units:
            new_unit = Unit(
                id=len(new_population),
                grid_x=5, grid_y=self.tile_map.grid_height // 2,
                brain=elite_unit.brain.clone(),
                tile_map=self.tile_map,
                whisker_length=self.whisker_length
            )
            new_population.append(new_unit)

        while len(new_population) < self.population_size:
            parent1 = fitness_scores[np.random.randint(0, self.population_size // 2)][0]
            parent2 = fitness_scores[np.random.randint(0, self.population_size // 2)][0]
            child_brain = MLP.crossover(parent1.brain, parent2.brain)
            child_brain.mutate(mutation_rate=mutation_rate)
            new_unit = Unit(
                id=len(new_population),
                grid_x=5, grid_y=self.tile_map.grid_height // 2,
                brain=child_brain,
                tile_map=self.tile_map,
                whisker_length=self.whisker_length
            )
            new_population.append(new_unit)

        self.population = new_population
        self.population_map = {unit.id: unit for unit in self.population}
        self.generation += 1
        self.clear_worker_caches()
        return fitness_scores[0][1]

    def clear_worker_caches(self):
        if self.pool and OPENCL_AVAILABLE:
            num_workers = self.pool._processes
            self.pool.map(clear_cache_worker, range(num_workers))

    def cleanup(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            print("Multiprocessing pool cleaned up.")
            self.pool = None
        if OPENCL_AVAILABLE:
            if self.perception_cache:
                for buf in self.perception_cache.values(): buf.release()
                self.perception_cache.clear()
            if self.tile_map_buf:
                self.tile_map_buf.release()
                self.tile_map_buf = None
            print("Released OpenCL buffers.")

    def rebuild_pool(self):
        self.cleanup()
        try:
            self.pool = Pool(processes=cpu_count(), initializer=init_worker, initargs=(self.tile_map,))
            print("Multiprocessing pool rebuilt successfully.")
        except (OSError, ImportError):
            self.pool = None
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

    def reset_simulation(self):
        self.population = self._create_initial_population()
        self.population_map = {unit.id: unit for unit in self.population}
        self.generation = 0
        self.clear_worker_caches()

    def set_population_size(self, new_size):
        self.population_size = new_size
        if self.num_to_draw > self.population_size:
            self.num_to_draw = self.population_size
        self.reset_simulation()
        if OPENCL_AVAILABLE:
            self._init_gpu_buffers()

    def save_fittest_brain(self, filepath_prefix="saved_brains/brain"):
        if not self.population: return
        fitness_scores = []
        for unit in self.population:
            distance = abs(unit.grid_x - self.target.grid_x) + abs(unit.grid_y - self.target.grid_y)
            max_dist = self.tile_map.grid_width + self.tile_map.grid_height
            fitness = (max_dist - distance) ** 2
            fitness_scores.append((unit, fitness))

        if not fitness_scores: return
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        fittest_unit = fitness_scores[0][0]
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        weights_path = f"{filepath_prefix}_weights.npz"
        np.savez(weights_path, *fittest_unit.brain.weights, *fittest_unit.brain.biases)
        print(f"Saved fittest brain to {weights_path}")

    def load_brain_from_file(self, filepath_prefix="saved_brains/brain"):
        weights_path = f"{filepath_prefix}_weights.npz"
        if not os.path.exists(weights_path):
            print(f"Error: Brain weights file not found at {weights_path}")
            return

        loaded_brain = MLP(self.mlp_architecture)
        with np.load(weights_path) as data:
            num_weight_matrices = len(loaded_brain.weights)
            for i in range(num_weight_matrices):
                loaded_brain.weights[i] = data[f'arr_{i}']
            for i in range(len(loaded_brain.biases)):
                loaded_brain.biases[i] = data[f'arr_{i + num_weight_matrices}']

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
