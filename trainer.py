import numpy as np
import json
import os
import pygame
from multiprocessing import Pool, cpu_count
from game import Unit, Target, Wall, Enemy, line_circle_intersection
from mlp import MLP
from map import Tile

# --- Worker Function for Multiprocessing ---

_tile_map_global = None

def init_worker(tile_map):
    """
    Initializer for each worker process.
    Makes the tile_map a global variable in the worker.
    """
    global _tile_map_global
    _tile_map_global = tile_map


def get_unit_inputs(unit_data, world_objects_data, target_pos_data):
    """
    A standalone version of the Unit.get_inputs method.
    This function is designed to be called from a worker process.
    It operates on simplified data structures (tuples, dicts) instead of full objects.
    """
    # Unpack data
    unit_pos = pygame.Vector2(unit_data['position'])
    unit_angle = unit_data['angle']
    num_whiskers = unit_data['num_whiskers']
    whisker_length = unit_data['whisker_length']
    perceivable_types = unit_data['perceivable_types']
    unit_velocity = pygame.Vector2(unit_data['velocity'])

    whisker_angles = np.linspace(-np.pi / 2, np.pi / 2, num_whiskers) if num_whiskers > 1 else np.array([0])

    # --- Whisker Calculations ---
    num_perceivables = len(perceivable_types)
    whisker_inputs = np.zeros((num_whiskers, num_perceivables))

    for i, whisker_angle in enumerate(whisker_angles):
        abs_angle = unit_angle + whisker_angle
        start_point = unit_pos
        end_point = unit_pos + pygame.Vector2(whisker_length, 0).rotate(np.rad2deg(abs_angle))

        closest_dist = whisker_length
        detected_type = None

        # Detect Walls from the global tile_map
        if "wall" in perceivable_types:
            dx, dy = end_point.x - start_point.x, end_point.y - start_point.y
            steps = max(abs(dx), abs(dy))
            if steps > 0:
                x_inc, y_inc = dx / steps, dy / steps
                x, y = start_point.x, start_point.y
                for _ in range(int(steps)):
                    grid_x = int(x // _tile_map_global.tile_size)
                    grid_y = int(y // _tile_map_global.tile_size)
                    if _tile_map_global.get_tile(grid_x, grid_y) == Tile.WALL:
                        closest_dist = unit_pos.distance_to(pygame.Vector2(x, y))
                        detected_type = "wall"
                        break
                    x += x_inc
                    y += y_inc

        # Detect other objects
        # Create a temporary list of all objects to check against, excluding the unit itself
        all_objects_data = world_objects_data + [d for d in unit_data.get('population_data', []) if d['id'] != unit_data['id']]

        for obj_data in all_objects_data:
            # Skip self-check - this is now redundant if population_data is filtered, but safe
            if obj_data['id'] == unit_data.get('id') and obj_data['type'] == 'unit':
                continue

            dist = line_circle_intersection(start_point, end_point, pygame.Vector2(obj_data['position']), obj_data['size'])
            if dist is not None and dist < closest_dist:
                closest_dist = dist
                detected_type = obj_data['type']

        if detected_type and detected_type in perceivable_types:
            type_index = perceivable_types.index(detected_type)
            # Clamp distance to avoid division by zero or negative values
            clamped_dist = max(0, min(closest_dist, whisker_length))
            whisker_inputs[i, type_index] = 1.0 - (clamped_dist / whisker_length)

    # --- Target Vector Calculation ---
    target_pos = pygame.Vector2(target_pos_data)
    relative_vec = target_pos - unit_pos
    relative_vec = relative_vec.rotate(-np.rad2deg(unit_angle))
    norm_dx = np.clip(relative_vec.x / _tile_map_global.pixel_width, -1, 1)
    norm_dy = np.clip(relative_vec.y / _tile_map_global.pixel_height, -1, 1)
    target_inputs = np.array([norm_dx, norm_dy])

    # --- Concatenate all inputs ---
    flat_whisker_inputs = whisker_inputs.flatten()
    other_inputs = np.array([unit_velocity.length() / 2.0, unit_angle / (2 * np.pi)]) # unit.speed is 2.0
    return np.concatenate((flat_whisker_inputs, other_inputs, target_inputs))


def run_single_unit_step(unit_data_bundle):
    """
    The main worker function. It takes a unit's data, calculates its next state,
    and returns the results.
    """
    # 1. Get inputs from the environment
    inputs = get_unit_inputs(
        unit_data_bundle['unit'],
        unit_data_bundle['world_objects'],
        unit_data_bundle['target_position']
    )

    # 2. Perform the forward pass of the neural network
    brain = unit_data_bundle['unit']['brain']
    actions = brain.forward(inputs)

    # 3. Return the calculated actions and the unit's ID
    return {
        "id": unit_data_bundle['unit']['id'],
        "actions": actions
    }


class TrainingMode:
    NAVIGATE = 1
    COMBAT = 2

class TrainingSimulation:
    """
    Manages the genetic algorithm training process.
    """
    def __init__(self, population_size, world_size, tile_map, num_whiskers=7, perceivable_types=None, whisker_length=150):
        self.population_size = population_size
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
        Runs a single step of the simulation for the entire population using a multiprocessing pool.
        """
        world_objects_data = [obj.to_dict() for obj in self.world_objects]
        target_pos_data = (self.target.position.x, self.target.position.y)
        population_data = [u.to_dict() for u in self.population]

        tasks = []
        for unit_dict in population_data:
            # The 'population_data' is included in each task bundle so that workers can
            # perceive other units.
            unit_dict['population_data'] = population_data
            tasks.append({
                'unit': unit_dict,
                'world_objects': world_objects_data,
                'target_position': target_pos_data
            })

        # Use the pool to distribute the work.
        # If the pool doesn't exist, fall back to a sequential map.
        if self.pool:
            results = self.pool.map(run_single_unit_step, tasks)
        else:
            # This ensures the program can still run even if the pool failed to init
            init_worker(self.tile_map)
            results = [run_single_unit_step(task) for task in tasks]

        for result in results:
            unit = self.population_map[result['id']]
            actions = result['actions']
            unit.update(actions, self.projectiles)
            unit.position.x = np.clip(unit.position.x, 0, self.world_width)
            unit.position.y = np.clip(unit.position.y, 0, self.world_height)

        # Update and check projectiles
        for proj in self.projectiles[:]: # Iterate over a copy
            proj.update()
            if proj.lifespan <= 0:
                self.projectiles.remove(proj)
                continue

            # Check for collision with enemy
            if proj.position.distance_to(self.enemy.position) < self.enemy.size:
                damage = 10
                self.enemy.health -= damage
                proj.owner.damage_dealt += damage
                self.projectiles.remove(proj)
                if self.enemy.health <= 0:
                    print("Enemy defeated!")
                    # For now, just reset its health for the next generation
                    self.enemy.health = 100

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
