import numpy as np
import json
import os
from game import Unit, Target, Wall, Enemy
from mlp import MLP

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

        # Define the MLP architecture
        # The number of inputs depends on whiskers, perceivable types, and now the target vector
        # whiskers * types + velocity + angle + target_dx + target_dy
        num_inputs = self.num_whiskers * len(self.perceivable_types) + 2 + 2
        self.mlp_architecture = [num_inputs, 16, 2]

        # Create world objects (dynamic objects, walls are handled by the map)
        self.target = Target(self.world_width - 50, self.world_height / 2)
        self.enemy = Enemy(self.world_width - 100, self.world_height / 2 + 100)
        self.world_objects = [self.target, self.enemy]
        self.projectiles = []

        # Create the initial population
        self.population = self._create_initial_population()

    def _create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            brain = MLP(self.mlp_architecture)
            unit = Unit(
                x=50, y=self.world_height / 2, brain=brain,
                num_whiskers=self.num_whiskers,
                whisker_length=self.whisker_length,
                perceivable_types=self.perceivable_types,
                tile_map=self.tile_map
            )
            population.append(unit)
        return population

    def run_generation_step(self):
        """
        Runs a single step of the simulation for the entire population.
        """
        # Update units
        for unit in self.population:
            inputs = unit.get_inputs(self.world_objects, self.target)
            actions = unit.brain.forward(inputs)
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
                # Fitness is primarily based on damage dealt.
                # A small bonus for being closer to the enemy.
                distance = unit.position.distance_to(self.enemy.position)
                fitness = unit.damage_dealt * 100 + (self.world_width - distance)
                fitness_scores.append((unit, fitness))

        # 2. Sort units by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Create the next generation
        new_population = []

        # Elitism: The best units carry over directly
        num_elites = int(self.population_size * elitism_frac)
        elite_units = [item[0] for item in fitness_scores[:num_elites]]
        for elite_unit in elite_units:
            # Create a new unit instance but keep the brain and configuration
            new_unit = Unit(
                x=50, y=self.world_height / 2, brain=elite_unit.brain,
                num_whiskers=self.num_whiskers, perceivable_types=self.perceivable_types,
                whisker_length=self.whisker_length, tile_map=self.tile_map
            )
            new_population.append(new_unit)

        # Crossover and Mutation
        while len(new_population) < self.population_size:
            # Select parents (simple selection: pick from the top half)
            parent1 = fitness_scores[np.random.randint(0, self.population_size // 2)][0]
            parent2 = fitness_scores[np.random.randint(0, self.population_size // 2)][0]

            # Create child
            child_brain = MLP.crossover(parent1.brain, parent2.brain)

            # Mutate child
            child_brain.mutate(mutation_rate=mutation_rate)

            # Add new unit with the child's brain to the population
            new_unit = Unit(
                x=50, y=self.world_height / 2, brain=child_brain,
                num_whiskers=self.num_whiskers, perceivable_types=self.perceivable_types,
                whisker_length=self.whisker_length, tile_map=self.tile_map
            )
            new_population.append(new_unit)

        self.population = new_population
        self.generation += 1

        # Return best fitness for logging
        return fitness_scores[0][1]

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
        self.projectiles = [] # Clear projectiles
        self.enemy.health = 100 # Reset enemy health
        for unit in self.population:
            unit.damage_dealt = 0
        self.generation = 0

    def save_fittest_brain(self, filepath_prefix="saved_brains/brain"):
        """
        Saves the architecture and weights of the fittest brain in the population,
        using the correct fitness function for the current training mode.
        """
        # Find the fittest unit by calculating fitness based on the current mode
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
            print("Warning: Could not determine fittest brain. No units or fitness scores.")
            return

        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        fittest_unit = fitness_scores[0][0]

        # Ensure the save directory exists
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)

        # Save architecture to JSON
        arch_data = {
            "layer_sizes": fittest_unit.brain.layer_sizes,
            "num_whiskers": fittest_unit.num_whiskers,
            "perceivable_types": fittest_unit.perceivable_types
        }
        json_path = f"{filepath_prefix}_arch.json"
        with open(json_path, 'w') as f:
            json.dump(arch_data, f, indent=4)

        # Save weights and biases to NPZ
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

        # Load architecture
        with open(json_path, 'r') as f:
            arch_data = json.load(f)

        layer_sizes = arch_data["layer_sizes"]
        num_whiskers = arch_data["num_whiskers"]
        perceivable_types = arch_data.get("perceivable_types", ["wall", "enemy", "unit"]) # Default for older saves

        # Create a new brain and load weights
        loaded_brain = MLP(layer_sizes)
        with np.load(weights_path) as data:
            num_weight_matrices = len(loaded_brain.weights)

            # First N files are weights, the rest are biases
            for i in range(num_weight_matrices):
                loaded_brain.weights[i] = data[f'arr_{i}']
            for i in range(len(loaded_brain.biases)):
                loaded_brain.biases[i] = data[f'arr_{i + num_weight_matrices}']

        # Rebuild the population with clones of the loaded brain
        self.rebuild_with_new_architecture(layer_sizes, num_whiskers, perceivable_types)
        for unit in self.population:
            unit.brain = loaded_brain # Assign the loaded brain to all units

        print(f"Loaded brain from {filepath_prefix} and rebuilt population.")
