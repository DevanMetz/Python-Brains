import unittest
from unittest.mock import patch
import numpy as np
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp_cupy import MLPCupy, CUPY_AVAILABLE
from mlp import MLP

# A decorator to skip tests if CuPy is not available
skip_if_no_cupy = unittest.skipIf(not CUPY_AVAILABLE, "CuPy is not installed or a GPU is not available")

class TestMLPCupy(unittest.TestCase):
    """
    Tests for the MLPCupy class.
    These tests cover the core functionality that relies on the NumPy backend
    (initialization, crossover, mutation), which should work regardless of
    CuPy's availability.
    """

    def test_initialization(self):
        """Test if the MLPCupy is initialized with correct layer, weight, and bias shapes."""
        arch = [2, 4, 1]
        mlp = MLPCupy(arch)
        self.assertEqual(len(mlp.weights), 2)
        self.assertEqual(mlp.weights[0].shape, (2, 4))
        self.assertEqual(mlp.weights[1].shape, (4, 1))
        self.assertEqual(len(mlp.biases), 2)
        self.assertEqual(mlp.biases[0].shape, (1, 4))
        self.assertEqual(mlp.biases[1].shape, (1, 1))
        self.assertIsInstance(mlp, MLP) # Should be a subclass of the original

    def test_crossover(self):
        """Test the crossover functionality (operates on NumPy arrays)."""
        arch = [3, 5, 2]
        parent1 = MLPCupy(arch)
        parent2 = MLPCupy(arch)
        child = MLPCupy.crossover(parent1, parent2)
        self.assertEqual(child.layer_sizes, arch)
        # High probability that the child's weights are different from both parents
        self.assertFalse(np.array_equal(child.weights[0], parent1.weights[0]))
        self.assertFalse(np.array_equal(child.weights[0], parent2.weights[0]))

    def test_mutation(self):
        """Test if mutation alters the weights (operates on NumPy arrays)."""
        arch = [4, 6, 3]
        mlp = MLPCupy(arch)
        original_weights = [w.copy() for w in mlp.weights]
        mlp.mutate(mutation_rate=1.0, mutation_amount=0.1) # 100% mutation rate
        self.assertFalse(np.array_equal(mlp.weights[0], original_weights[0]))

@skip_if_no_cupy
class TestMLPCupyForwardPass(unittest.TestCase):
    """
    Tests specifically for the forward pass, which can use the GPU.
    This entire class will be skipped if CuPy is not available.
    """

    def test_forward_pass_gpu(self):
        """Test the forward pass on the GPU for correct output shape and value range."""
        arch = [5, 10, 3]
        mlp = MLPCupy(arch)
        inputs = np.random.randn(1, 5)
        output = mlp.forward(inputs)
        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1), "Output should be in [-1, 1] for tanh")
        # The output should be a NumPy array, not a CuPy array
        self.assertIsInstance(output, np.ndarray)

    def test_forward_pass_with_1d_input(self):
        """Test the forward pass with a 1D input array."""
        arch = [10, 5, 2]
        mlp = MLPCupy(arch)
        inputs = np.random.randn(10) # 1D input
        output = mlp.forward(inputs)
        self.assertEqual(output.shape, (1, 2))
        self.assertIsInstance(output, np.ndarray)


class TestMLPCupyCPUFallback(unittest.TestCase):
    """
    Tests the CPU fallback functionality by explicitly disabling CuPy.
    """

    def test_forward_pass_cpu_fallback(self):
        """Test the forward pass CPU fallback by mocking CUPY_AVAILABLE."""
        # Import the module itself so we can monkeypatch the global variable
        import mlp_cupy
        original_cupy_available = mlp_cupy.CUPY_AVAILABLE
        mlp_cupy.CUPY_AVAILABLE = False # Force fallback

        try:
            arch = [5, 10, 3]
            mlp = MLPCupy(arch)
            inputs = np.random.randn(1, 5)

            # This should now call the original MLP.forward()
            output = mlp.forward(inputs)

            self.assertEqual(output.shape, (1, 3))
            self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
            self.assertIsInstance(output, np.ndarray)
        finally:
            # IMPORTANT: Restore the original value to not affect other tests
            mlp_cupy.CUPY_AVAILABLE = original_cupy_available

class TestMLPCupyGPUMocked(unittest.TestCase):
    """
    Tests the GPU functionality by mocking CuPy calls.
    """

    @patch('mlp_cupy.CUPY_AVAILABLE', True)
    @patch('cupy.asarray')
    def test_forward_pass_gpu_failure_fallback(self, mock_asarray):
        """Test that a CuPy error during forward pass triggers the CPU fallback."""
        # Configure the mock to raise an exception when called
        mock_asarray.side_effect = RuntimeError("Simulated CuPy Error")

        arch = [5, 10, 3]
        mlp = MLPCupy(arch)
        inputs = np.random.randn(1, 5)

        # This forward pass should fail on the GPU and fall back to the CPU
        output = mlp.forward(inputs)

        # Verify that the output is what we'd expect from the CPU implementation
        self.assertEqual(output.shape, (1, 3))
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
        self.assertIsInstance(output, np.ndarray)

    @patch('mlp_cupy.CUPY_AVAILABLE', True)
    @patch('cupy.asnumpy')
    @patch('cupy.tanh')
    @patch('cupy.asarray')
    def test_forward_pass_gpu_success_mocked(self, mock_asarray, mock_tanh, mock_asnumpy):
        """Test the GPU forward pass with mocked CuPy functions."""
        # Make asarray and asnumpy return the data as is (acting as placeholders)
        mock_asarray.side_effect = lambda x: np.array(x) if not isinstance(x, np.ndarray) else x
        mock_asnumpy.side_effect = lambda x: x
        # Mock tanh to behave like numpy's tanh for consistent results
        mock_tanh.side_effect = np.tanh

        arch = [2, 3, 1]
        mlp = MLPCupy(arch)
        # Use a list input to test the conversion to a NumPy array
        inputs = [0.5, 0.5]
        output = mlp.forward(inputs)

        self.assertEqual(output.shape, (1, 1))
        self.assertIsInstance(output, np.ndarray)

        # Check that the GPU path was taken by checking mock calls
        self.assertTrue(mock_asarray.called)
        self.assertTrue(mock_tanh.called)
        self.assertTrue(mock_asnumpy.called)


if __name__ == '__main__':
    unittest.main()
