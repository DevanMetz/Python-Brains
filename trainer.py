import numpy as np
from game import Unit, Target, Wall
from mlp import MLP

class TrainingSimulation:
    """
    Manages the genetic algorithm training process.
    """
    def __init__(self, population_size=50, world_size=(800, 600)):
        self.population_size = population_size
        self.world_width, self.world_height = world_size
        self.generation = 0

        # Define the MLP architecture
        # Inputs: 7 whiskers + velocity + angle = 9
        # Outputs: turn + move = 2
        self.mlp_architecture = [9, 16, 2]

        # Create world objects
        self.target = Target(self.world_width - 50, self.world_height / 2)
        self.world_objects = [self.target]
        # Add some boundary walls
        self.world_objects.append(Wall(0, 0, self.world_width, 10)) # Top
        self.world_objects.append(Wall(0, self.world_height - 10, self.world_width, 10)) # Bottom
        self.world_objects.append(Wall(0, 0, 10, self.world_height)) # Left
        self.world_objects.append(Wall(self.world_width - 10, 0, 10, self.world_height)) # Right

        # Create the initial population
        self.population = self._create_initial_population()

    def _create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            brain = MLP(self.mlp_architecture)
            unit = Unit(x=50, y=self.world_height / 2, brain=brain)
            population.append(unit)
        return population

    def run_generation_step(self):
        """
        Runs a single step of the simulation for the entire population.
        """
        for unit in self.population:
            # 1. Get inputs from the world
            inputs = unit.get_inputs(self.world_objects)

            # 2. Get actions from the MLP brain
            actions = unit.brain.forward(inputs)

            # 3. Update the unit's state
            unit.update(actions)

            # Optional: Keep units within bounds
            unit.position.x = np.clip(unit.position.x, 0, self.world_width)
            unit.position.y = np.clip(unit.position.y, 0, self.world_height)

    def evolve_population(self, elitism_frac=0.1, mutation_rate=0.05):
        """
        Evaluates fitness and creates a new generation.
        """
        # 1. Evaluate fitness
        fitness_scores = []
        for unit in self.population:
            # Fitness is the inverse of the distance to the target. Closer is better.
            distance = unit.position.distance_to(self.target.position)
            fitness = (self.world_width - distance) ** 2 # Squaring gives more weight to closer units
            fitness_scores.append((unit, fitness))

        # 2. Sort units by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Create the next generation
        new_population = []

        # Elitism: The best units carry over directly
        num_elites = int(self.population_size * elitism_frac)
        elite_units = [item[0] for item in fitness_scores[:num_elites]]
        for elite_unit in elite_units:
            # Create a new unit instance but keep the brain
            new_unit = Unit(x=50, y=self.world_height / 2, brain=elite_unit.brain)
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
            new_unit = Unit(x=50, y=self.world_height / 2, brain=child_brain)
            new_population.append(new_unit)

        self.population = new_population
        self.generation += 1

        # Return best fitness for logging
        return fitness_scores[0][1]
