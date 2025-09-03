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
    from math_utils_opencl import opencl_vectorized_line_circle_intersection

# --- Worker Function for Multiprocessing ---

_tile_map_global = None
_gpu_brain_cache = {}

def init_worker(tile_map):
    """Initializer for each worker process in the multiprocessing pool.

    This function sets up global variables within the worker's scope.
    This is an optimization to avoid serializing and sending large, static
    data with every task.

    Args:
        tile_map (TileMap): The global tile map for the simulation.
    """
    global _tile_map_global, _gpu_brain_cache
    _tile_map_global = tile_map
    # The brain cache is also made global to persist across tasks within
    # the same worker process. It stores GPU-resident copies of brain weights.
    _gpu_brain_cache = {}


def get_unit_inputs(unit_data, local_objects_data, target_pos_data):
    """Calculates all sensory inputs for a single unit's MLP brain.

    This function is a critical part of the simulation, responsible for
    translating the world state into a numerical format that the MLP can
    understand. It performs the following steps:
    1.  Casts "whiskers" (rays) out from the unit.
    2.  Uses a high-performance vectorized method (GPU-accelerated with CuPy
        if available) to find intersections between these whiskers and nearby
        game objects.
    3.  Performs a separate check for whisker intersections with wall tiles.
    4.  Combines these results to find the closest object detected by each
        whisker.
    5.  Encodes the distance and type of the detected object into a normalized
        input value.
    6.  Calculates the unit's relative position to the target.
    7.  Concatenates whisker inputs, target vector, and the unit's internal
        state (velocity, angle) into a single input vector for the MLP.

    A fallback to a simpler, iterative method is included if CuPy is not
    available or if the vectorized calculation fails.

    Args:
        unit_data (dict): A dictionary containing the serialized state of the
            unit being processed.
        local_objects_data (list[dict]): A list of serialized dictionaries for
            game objects that are near the unit (pre-filtered by the Quadtree).
        target_pos_data (tuple[float, float]): The (x, y) coordinates of the
            current target object.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The final, flattened numpy array of all MLP inputs.
            - list[dict]: A list of dictionaries containing debug information
              for each whisker, used for visualization.
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

    # Use the OPENCL_AVAILABLE flag to choose the execution path.
    if OPENCL_AVAILABLE:
        try:
            # --- GPU-accelerated vectorized path (OpenCL) ---
            abs_angles = unit_angle + whisker_angles

            # Prepare whisker start and end points as NumPy arrays
            start_points_np = np.tile(unit_pos, (num_whiskers, 1))
            end_vectors_np = np.column_stack([
                np.cos(abs_angles) * whisker_length,
                np.sin(abs_angles) * whisker_length
            ])
            end_points_np = start_points_np + end_vectors_np

            # Prepare circle data as NumPy arrays
            if local_objects_data:
                centers_np = np.array([d['position'] for d in local_objects_data], dtype=np.float32)
                radii_np = np.array([d['size'] for d in local_objects_data], dtype=np.float32)
            else:
                centers_np = np.empty((0, 2), dtype=np.float32)
                radii_np = np.empty(0, dtype=np.float32)

            # Call the OpenCL kernel wrapper
            obj_distances, obj_indices = opencl_vectorized_line_circle_intersection(
                start_points_np, end_points_np, centers_np, radii_np
            )

            # --- Wall detection (remains on CPU as it's iterative and complex to vectorize) ---
            wall_distances = np.full(num_whiskers, np.inf, dtype=np.float32)
            if "wall" in perceivable_types:
                for i, whisker_angle in enumerate(whisker_angles):
                    # This logic is duplicated from the fallback path for simplicity
                    abs_angle_rad = unit_angle + whisker_angle
                    start_point = unit_pos
                    end_point = start_point + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(abs_angle_rad))
                    dx, dy = end_point.x - start_point.x, end_point.y - start_point.y
                    steps = int(max(abs(dx), abs(dy)))
                    if steps > 0:
                        x_inc, y_inc = dx / steps, dy / steps
                        for j in range(steps):
                            x, y = start_point.x + j * x_inc, start_point.y + j * y_inc
                            if _tile_map_global.get_tile_at_pixel(x, y) == Tile.WALL:
                                wall_distances[i] = unit_pos.distance_to(pygame.Vector2(x, y))
                                break

            # --- Combine GPU (object) and CPU (wall) results ---
            final_distances = np.where(obj_distances < wall_distances, obj_distances, wall_distances)
            final_indices = np.where(obj_distances < wall_distances, obj_indices, -2) # -2 for wall

            # --- Final processing to generate inputs and debug info ---
            object_types = [d['type'] for d in local_objects_data]
            for i in range(num_whiskers):
                dist, idx = final_distances[i], final_indices[i]
                detected_type = None
                abs_angle_rad = unit_angle + whisker_angles[i]
                start_point = unit_pos
                full_end_point = start_point + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(abs_angle_rad))
                intersect_point = full_end_point

                if dist != np.inf:
                    detected_type = "wall" if idx == -2 else object_types[idx]
                    intersect_point = start_point + pygame.Vector2(dist, 0).rotate(np.rad2deg(abs_angle_rad))
                    if detected_type and detected_type in perceivable_types:
                        type_index = perceivable_types.index(detected_type)
                        clamped_dist = max(0, min(dist, whisker_length))
                        whisker_inputs[i, type_index] = 1.0 - (clamped_dist / whisker_length)

                whisker_debug_info.append({'start': (start_point.x, start_point.y), 'end': (intersect_point.x, intersect_point.y), 'full_end': (full_end_point.x, full_end_point.y), 'type': detected_type})

        except Exception as e:
            print(f"Warning: OpenCL-based perception failed with error: {e}. Falling back to CPU.")
            # This will cause the next check to fail and use the CPU path for this unit.
            # We do NOT assign to OPENCL_AVAILABLE here to avoid the UnboundLocalError.
            pass

    if not OPENCL_AVAILABLE:
        # --- Fallback to original iterative method ---
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


def run_single_unit_step(unit_data, local_objects_data, target_position_data):
    """The main worker function for processing one unit for one simulation step.

    This function is executed by a worker in the multiprocessing pool. It takes
    the serialized state of a unit and its local environment, calculates the
    necessary inputs for the unit's brain, performs a forward pass on the MLP
    to get the unit's actions, and returns the results.

    Args:
        unit_data (dict): Serialized state of the unit.
        local_objects_data (list[dict]): Serialized list of nearby objects.
        target_position_data (tuple[float, float]): Position of the target.

    Returns:
        dict: A dictionary containing the results of the step, including the
            unit's ID, the actions decided by its brain, and whisker
            visualization data.
    """
    inputs, whisker_debug_info = get_unit_inputs(
        unit_data,
        local_objects_data,
        target_position_data
    )

    brain = unit_data['brain']

    # --- Optimized MLP Forward Pass with OpenCL Caching ---
    actions = None
    if OPENCL_AVAILABLE:
        try:
            from mlp_opencl import context # Get the worker's context

            # Use the memory address of the weights array as a unique ID.
            brain_id = brain.weights[0].ctypes.data

            # Check if this brain's OpenCL buffers are already cached.
            if brain_id not in _gpu_brain_cache:
                # If not, create, transfer, and cache the buffers.
                weights_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w.astype(np.float32)) for w in brain.weights]
                biases_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.astype(np.float32)) for b in brain.biases]
                _gpu_brain_cache[brain_id] = {'weights': weights_bufs, 'biases': biases_bufs}

            # Run the forward pass using the cached OpenCL buffers.
            cached_buffers = _gpu_brain_cache[brain_id]
            actions = brain.forward(inputs, cached_buffers=cached_buffers)

        except Exception as e:
            # Fallback to default forward if caching or GPU execution fails.
            print(f"OpenCL forward pass with caching failed: {e}. Falling back.")
            actions = brain.forward(inputs) # This will call the NumPy version
    else:
        # If OpenCL is not available, use the standard NumPy forward method.
        actions = brain.forward(inputs)


    return {
        "id": unit_data['id'],
        "actions": actions,
        "whisker_debug_info": whisker_debug_info
    }


def clear_cache_worker(_):
    """
    A worker task to clear the global GPU brain cache.

    This function iterates through the cached brain buffers (weights and biases)
    and explicitly calls .release() on each OpenCL buffer to free the
    corresponding memory on the GPU device before clearing the cache dictionary.
    """
    global _gpu_brain_cache
    if _gpu_brain_cache:
        for brain_data in _gpu_brain_cache.values():
            for buf in brain_data['weights']:
                buf.release()
            for buf in brain_data['biases']:
                buf.release()
        _gpu_brain_cache.clear()
    return True


def run_single_unit_step_wrapper(task_data):
    """Unpacks a task dictionary and calls the main worker function.

    This wrapper is necessary because `pool.map` only accepts functions that
    take a single argument. This function takes a dictionary containing all the
    necessary data and unpacks it into the arguments required by
    `run_single_unit_step`.

    Args:
        task_data (dict): A dictionary containing the arguments for the main
            worker function: `unit_data`, `local_objects_data`, and
            `target_position_data`.

    Returns:
        dict: The result from `run_single_unit_step`.
    """
    return run_single_unit_step(
        task_data['unit_data'],
        task_data['local_objects_data'],
        task_data['target_position_data']
    )


class TrainingMode:
    """An enumeration for the different training modes available."""
    NAVIGATE = 1
    """Fitness is based on proximity to a target."""
    COMBAT = 2
    """Fitness is based on damage dealt to an enemy."""

class TrainingSimulation:
    """
    Manages the genetic algorithm training process for a population of units.

    This class is the central controller for the simulation. It initializes the
    world, creates and manages a population of AI-controlled units, and runs
    the evolutionary training loop. It uses a multiprocessing pool to parallelize
    the simulation of each unit and a Quadtree to optimize collision and
    perception checks.

    Attributes:
        population_size (int): The total number of units in the simulation.
        num_to_draw (int): The number of units to render on screen.
        world_width (int): The width of the simulation world in pixels.
        world_height (int): The height of the simulation world in pixels.
        tile_map (TileMap): The map of wall and empty tiles.
        generation (int): The current generation number.
        num_whiskers (int): The number of sensory whiskers each unit has.
        whisker_length (float): The maximum length of each whisker.
        perceivable_types (list[str]): The object types units can "see".
        training_mode (TrainingMode): The current fitness evaluation mode.
        pool (multiprocessing.Pool): The pool of worker processes.
        mlp_architecture (list[int]): The layer sizes of the units' brains.
        target (Target): The target object for navigation tasks.
        enemy (Enemy): The enemy object for combat tasks.
        world_objects (list): A list of static objects in the world.
        projectiles (list[Projectile]): A list of active projectiles.
        population (list[Unit]): The list of all units in the simulation.
        population_map (dict[int, Unit]): A mapping from unit ID to unit object.
        quadtree (QuadTree): The spatial partitioning data structure.
    """
    def __init__(self, population_size, world_size, tile_map, num_whiskers=7, perceivable_types=None, whisker_length=150):
        """Initializes the training simulation environment.

        Args:
            population_size (int): The number of units in the population.
            world_size (tuple[int, int]): The (width, height) of the simulation
                world in pixels.
            tile_map (TileMap): The tile map object for the simulation.
            num_whiskers (int, optional): The number of whiskers per unit.
                Defaults to 7.
            perceivable_types (list[str], optional): A list of object type
                strings that units can perceive. Defaults to a standard set.
            whisker_length (int, optional): The maximum length of the whiskers.
                Defaults to 150.
        """
        self.population_size = population_size
        self.num_to_draw = population_size # Default to drawing all units
        self.world_width, self.world_height = world_size
        self.tile_map = tile_map
        self.generation = 0
        self.num_whiskers = num_whiskers
        self.whisker_length = whisker_length
        self.perceivable_types = perceivable_types if perceivable_types is not None else ["wall", "enemy", "unit"]
        self.training_mode = TrainingMode.NAVIGATE

        # --- GPU Status Warning ---
        if not OPENCL_AVAILABLE:
            print("\n" + "="*60)
            print(" WARNING: GPU ACCELERATION NOT AVAILABLE ".center(60, "="))
            print("="*60)
            print(" PyOpenCL not found or no compatible GPU device detected.")
            print(" The simulation will run in a slower, CPU-only mode.")
            print(" To enable GPU acceleration:")
            print(" 1. Ensure you have a compatible GPU with OpenCL drivers.")
            print("    (These are typically included with standard GPU drivers).")
            print(" 2. Install the 'pyopencl' package: pip install pyopencl")
            print("="*60 + "\n")


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
        """Creates the first generation of units with random brains.

        Returns:
            list[Unit]: A list of newly created Unit objects.
        """
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
        """Runs a single frame/step of the simulation for the entire population.

        This method orchestrates the core simulation logic for one step:
        1.  Rebuilds the Quadtree with the current positions of all objects.
        2.  For each unit, queries the Quadtree to find nearby objects.
        3.  Creates a task for each unit containing its state and local object
            data.
        4.  Distributes these tasks to the multiprocessing pool for parallel
            execution.
        5.  Applies the results (actions) from the workers back to the units.
        6.  Updates all projectiles and handles their collisions.
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
        if self.pool and CUPY_AVAILABLE:
            # The argument to map is an iterable of inputs for the function.
            # We just need to run the function once for each worker.
            num_workers = self.pool._processes
            self.pool.map(clear_cache_worker, range(num_workers))
            # print("Cleared GPU caches in worker processes.") # Optional: for debugging

    def cleanup(self):
        """
        Cleans up resources, specifically terminating the multiprocessing pool.
        This should be called before the application exits.
        """
        if self.pool:
            self.pool.close()
            self.pool.join()
            print("Multiprocessing pool cleaned up.")
            self.pool = None

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
