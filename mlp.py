import numpy as np

class MLP:
    """
    A simple Multilayer Perceptron class.
    """

    def __init__(self, layer_sizes):
        """
        Initializes the MLP.

        Args:
            layer_sizes (list of int): A list containing the number of neurons in each layer.
                                       Example: [2, 4, 1] for 2 inputs, 4 hidden neurons, 1 output.
        """
        self.layer_sizes = layer_sizes

        # Initialize weights and biases
        # Weights are matrices connecting layer i to layer i+1
        # Biases are vectors for each neuron in layer i+1
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # Weights initialized with random values for symmetry breaking
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            self.weights.append(weight_matrix)

            # Biases initialized to zero
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.biases.append(bias_vector)

    @staticmethod
    def _tanh(x):
        """The hyperbolic tangent activation function."""
        return np.tanh(x)

    def forward(self, inputs):
        """
        Performs a forward pass through the network.

        Args:
            inputs (np.ndarray): The input vector for the network.

        Returns:
            np.ndarray: The output vector from the network.
        """
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)

        # Ensure input is a 2D array (batch of 1)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        current_layer_output = inputs

        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_layer_output, self.weights[i]) + self.biases[i]

            # Apply activation function
            # For the final layer, we might want a different activation,
            # but for now, tanh is fine for all layers.
            current_layer_output = self._tanh(z)

        return current_layer_output

    @staticmethod
    def crossover(parent1, parent2):
        """
        Creates a new child MLP by crossing over the weights and biases of two parents.

        Args:
            parent1 (MLP): The first parent.
            parent2 (MLP): The second parent.

        Returns:
            MLP: A new child MLP.
        """
        if parent1.layer_sizes != parent2.layer_sizes:
            raise ValueError("Parents must have the same layer sizes for crossover.")

        child = MLP(parent1.layer_sizes)

        # Crossover weights
        for i in range(len(parent1.weights)):
            crossover_point = np.random.randint(0, parent1.weights[i].size)
            p1_flat = parent1.weights[i].flatten()
            p2_flat = parent2.weights[i].flatten()
            child_flat = np.concatenate((p1_flat[:crossover_point], p2_flat[crossover_point:]))
            child.weights[i] = child_flat.reshape(parent1.weights[i].shape)

        # Crossover biases
        for i in range(len(parent1.biases)):
            crossover_point = np.random.randint(0, parent1.biases[i].size)
            p1_flat = parent1.biases[i].flatten()
            p2_flat = parent2.biases[i].flatten()
            child_flat = np.concatenate((p1_flat[:crossover_point], p2_flat[crossover_point:]))
            child.biases[i] = child_flat.reshape(parent1.biases[i].shape)

        return child

    def mutate(self, mutation_rate=0.01, mutation_amount=0.1):
        """
        Randomly mutates the weights and biases of the MLP.

        Args:
            mutation_rate (float): The probability of any given weight/bias being mutated.
            mutation_amount (float): The maximum amount to change a weight/bias by.
        """
        for i in range(len(self.weights)):
            mutation_mask = np.random.random(self.weights[i].shape) < mutation_rate
            random_mutation = (np.random.randn(*self.weights[i].shape) * mutation_amount)
            self.weights[i] += random_mutation * mutation_mask

        for i in range(len(self.biases)):
            mutation_mask = np.random.random(self.biases[i].shape) < mutation_rate
            random_mutation = (np.random.randn(*self.biases[i].shape) * mutation_amount)
            self.biases[i] += random_mutation * mutation_mask
