import unittest
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp import MLP

class TestMLP(unittest.TestCase):

    def test_initialization(self):
        """Test if the MLP is initialized with correct layer, weight, and bias shapes."""
        arch = [2, 4, 1]
        mlp = MLP(arch)

        self.assertEqual(len(mlp.weights), 2) # 2 sets of weights for 3 layers
        self.assertEqual(mlp.weights[0].shape, (2, 4))
        self.assertEqual(mlp.weights[1].shape, (4, 1))

        self.assertEqual(len(mlp.biases), 2) # 2 sets of biases
        self.assertEqual(mlp.biases[0].shape, (1, 4))
        self.assertEqual(mlp.biases[1].shape, (1, 1))

    def test_forward_pass(self):
        """Test the forward pass for correct output shape and value range."""
        arch = [5, 10, 3]
        mlp = MLP(arch)
        inputs = np.random.randn(1, 5)

        output = mlp.forward(inputs)

        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1), "Output should be in [-1, 1] for tanh")

    def test_crossover(self):
        """Test the crossover functionality."""
        arch = [3, 5, 2]
        parent1 = MLP(arch)
        parent2 = MLP(arch)

        child = MLP.crossover(parent1, parent2)

        self.assertEqual(child.layer_sizes, arch)
        # Check that child's weights are not the same as parent1's (high probability)
        self.assertFalse(np.array_equal(child.weights[0], parent1.weights[0]))

    def test_crossover_mismatched_arch(self):
        """Test that crossover raises an error for mismatched architectures."""
        parent1 = MLP([2, 3, 2])
        parent2 = MLP([2, 4, 2])

        with self.assertRaises(ValueError):
            MLP.crossover(parent1, parent2)

    def test_mutation(self):
        """Test if mutation alters the weights."""
        arch = [4, 6, 3]
        mlp = MLP(arch)

        # Create a deep copy of weights to compare against
        original_weights = [w.copy() for w in mlp.weights]

        mlp.mutate(mutation_rate=1.0, mutation_amount=0.1) # 100% mutation rate

        # Check that weights have changed
        self.assertFalse(np.array_equal(mlp.weights[0], original_weights[0]))

if __name__ == '__main__':
    unittest.main()
