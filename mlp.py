"""
Provides a simple, NumPy-based implementation of a Multilayer Perceptron (MLP).

This module contains the `MLP` class, which serves as the "brain" for the AI
units. It includes methods for initialization, forward propagation, and the
genetic algorithm operations of crossover and mutation.
"""
import numpy as np

class MLP:
    """A simple Multilayer Perceptron (MLP) implemented with NumPy.

    This class creates a fully connected neural network. It initializes weights
    and biases, and provides methods for a forward pass (inference) and for
    the genetic algorithm operations used in training.

    Attributes:
        layer_sizes (list[int]): The number of neurons in each layer, starting
            with the input layer and ending with the output layer.
        weights (list[np.ndarray]): A list of weight matrices, where each
            matrix connects one layer to the next.
        biases (list[np.ndarray]): A list of bias vectors, one for each layer
            (excluding the input layer).
    """

    def __init__(self, layer_sizes):
        """Initializes the MLP with a given architecture.

        Weights are initialized using He initialization, which is well-suited
        for layers with ReLU-like activation functions (tanh is similar).
        Biases are initialized to zero.

        Args:
            layer_sizes (list[int]): A list containing the number of neurons
                in each layer. Example: `[8, 16, 2]` for 8 inputs, 16 hidden
                neurons, and 2 outputs.
        """
        self.layer_sizes = layer_sizes

        # Initialize weights and biases
        # Weights are matrices connecting layer i to layer i+1
        # Biases are vectors for each neuron in layer i+1
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            # Weights initialized with He initialization for better training
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            self.weights.append(weight_matrix)

            # Biases initialized to zero
            bias_vector = np.zeros((1, layer_sizes[i+1]))
            self.biases.append(bias_vector)

    @staticmethod
    def _tanh(x):
        """The hyperbolic tangent activation function.

        This function is used to introduce non-linearity into the network,
        allowing it to learn more complex patterns. It squashes the output
        of each neuron to a range between -1 and 1.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The element-wise hyperbolic tangent of the input.
        """
        return np.tanh(x)

    def forward(self, inputs):
        """Performs a forward pass through the network to get an output.

        The input is propagated through each layer of the network. At each
        layer, a linear transformation (dot product with weights plus bias)
        is followed by the application of the tanh activation function.

        Args:
            inputs (np.ndarray): The input vector for the network. It should
                have a shape of (n,) or (1, n) where n is the number of input
                neurons.

        Returns:
            np.ndarray: The output vector from the network, with shape (1, m)
            where m is the number of output neurons.
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

            # Apply activation function for all layers
            current_layer_output = self._tanh(z)

        return current_layer_output

    @staticmethod
    def crossover(parent1, parent2):
        """Creates a new child MLP by crossing over two parent MLPs.

        This method implements a simple single-point crossover for each weight
        matrix and bias vector in the network. A random crossover point is
        chosen, and the child's parameters are created by taking the first
        part from `parent1` and the second part from `parent2`.

        Args:
            parent1 (MLP): The first parent MLP.
            parent2 (MLP): The second parent MLP.

        Returns:
            MLP: A new child MLP created from the two parents.

        Raises:
            ValueError: If the parents do not have the same layer sizes.
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

    def clone(self):
        """Creates a deep copy of the MLP instance.

        This is essential for the genetic algorithm, particularly for elitism,
        where the best-performing individuals need to be copied without
        modification to the next generation.

        Returns:
            MLP: A new MLP instance with identical weights and biases.
        """
        cloned_mlp = MLP(self.layer_sizes)
        cloned_mlp.weights = [w.copy() for w in self.weights]
        cloned_mlp.biases = [b.copy() for b in self.biases]
        return cloned_mlp

    def mutate(self, mutation_rate=0.01, mutation_amount=0.1):
        """Randomly mutates the weights and biases of the MLP in-place.

        This method iterates through all weights and biases. Each individual
        parameter has a `mutation_rate` chance of being mutated. If selected,
        a random value drawn from a normal distribution (scaled by
        `mutation_amount`) is added to the parameter.

        Args:
            mutation_rate (float, optional): The probability of any given
                weight or bias being mutated. Defaults to 0.01.
            mutation_amount (float, optional): The standard deviation of the
                normal distribution from which mutation values are drawn.
                Controls the magnitude of mutations. Defaults to 0.1.
        """
        for i in range(len(self.weights)):
            mutation_mask = np.random.random(self.weights[i].shape) < mutation_rate
            random_mutation = (np.random.randn(*self.weights[i].shape) * mutation_amount)
            self.weights[i] += random_mutation * mutation_mask

        for i in range(len(self.biases)):
            mutation_mask = np.random.random(self.biases[i].shape) < mutation_rate
            random_mutation = (np.random.randn(*self.biases[i].shape) * mutation_amount)
            self.biases[i] += random_mutation * mutation_mask
