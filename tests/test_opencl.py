import pytest
import numpy as np

# Conditional import of OpenCL modules to allow tests to be collected even if pyopencl is not installed
try:
    from mlp_opencl import MLPOpenCL, OPENCL_AVAILABLE
    from math_utils_opencl import opencl_vectorized_line_circle_intersection
except ImportError:
    OPENCL_AVAILABLE = False

from mlp import MLP
from math_utils import iterative_line_circle_intersection

# A decorator to skip tests if OpenCL is not available
# This prevents test failures in environments without a GPU or OpenCL drivers.
skip_if_no_opencl = pytest.mark.skipif(not OPENCL_AVAILABLE, reason="PyOpenCL not available or no compatible GPU found.")

@skip_if_no_opencl
class TestMLPOpenCL:
    def test_forward_pass_correctness(self):
        """
        Tests if the OpenCL MLP's forward pass produces the same result as the NumPy MLP.
        """
        # Define a network architecture
        layer_sizes = [10, 20, 5]

        # Create a standard NumPy-based MLP
        numpy_mlp = MLP(layer_sizes)

        # Create an OpenCL-based MLP
        opencl_mlp = MLPOpenCL(layer_sizes)

        # Manually copy the weights and biases from the numpy MLP to the OpenCL MLP
        # to ensure they are identical for a fair comparison.
        opencl_mlp.weights = [w.copy() for w in numpy_mlp.weights]
        opencl_mlp.biases = [b.copy() for b in numpy_mlp.biases]

        # Create a random input vector
        inputs = np.random.rand(1, layer_sizes[0]).astype(np.float32)

        # Get the output from both MLPs
        numpy_output = numpy_mlp.forward(inputs)
        opencl_output = opencl_mlp.forward(inputs)

        # Compare the results. They should be very close.
        # atol (absolute tolerance) is used to account for minor floating point differences.
        assert np.allclose(numpy_output, opencl_output, atol=1e-6), "OpenCL MLP output does not match NumPy MLP output."

    def test_caching_correctness(self):
        """
        Tests if the caching mechanism for OpenCL buffers works and produces correct results.
        """
        from mlp_opencl import context
        import pyopencl as cl

        layer_sizes = [8, 4]
        opencl_mlp = MLPOpenCL(layer_sizes)
        inputs = np.random.rand(1, layer_sizes[0]).astype(np.float32)

        # Manually create and cache the buffers
        weights_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w.astype(np.float32)) for w in opencl_mlp.weights]
        biases_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.astype(np.float32)) for b in opencl_mlp.biases]
        cached_buffers = {'weights': weights_bufs, 'biases': biases_bufs}

        # Run the forward pass with the cached buffers
        cached_output = opencl_mlp.forward(inputs, cached_buffers=cached_buffers)

        # Run the forward pass without caching (should produce the same result)
        uncached_output = opencl_mlp.forward(inputs)

        # Release the manually created buffers
        for buf_list in cached_buffers.values():
            for buf in buf_list:
                buf.release()

        assert np.allclose(cached_output, uncached_output, atol=1e-6), "Cached and uncached OpenCL outputs do not match."

    def test_cloned_instance_forward_pass(self):
        """
        Tests that a cloned MLPOpenCL instance can still perform a forward pass.
        This specifically targets the bug where `clone()` created a base MLP instance.
        """
        layer_sizes = [5, 5]
        opencl_mlp = MLPOpenCL(layer_sizes)
        inputs = np.random.rand(1, layer_sizes[0]).astype(np.float32)

        # Clone the object
        cloned_mlp = opencl_mlp.clone()

        # The cloned object should be of the same type
        assert isinstance(cloned_mlp, MLPOpenCL)

        # Calling forward with caching should now work without a TypeError
        try:
            # We don't need to check for correctness here, just that it doesn't crash.
            # The other tests handle correctness.
            cloned_mlp.forward(inputs, cached_buffers=None)
        except TypeError as e:
            pytest.fail(f"Cloned MLPOpenCL instance failed forward pass with TypeError: {e}")


@skip_if_no_opencl
class TestOpenCLMath:
    def test_intersection_correctness(self):
        """
        Tests if the OpenCL line-circle intersection kernel produces the same
        result as the iterative NumPy-based implementation.
        """
        # --- Test Case 1: Simple intersection ---
        p1s = np.array([[0, 0]], dtype=np.float32)
        p2s = np.array([[10, 0]], dtype=np.float32)
        centers = np.array([[5, 0]], dtype=np.float32)
        radii = np.array([1], dtype=np.float32)

        # Run OpenCL version
        dist_cl, idx_cl = opencl_vectorized_line_circle_intersection(p1s, p2s, centers, radii)

        # Compare with iterative version (which we trust as the ground truth)
        dist_iter = iterative_line_circle_intersection(p1s[0], p2s[0], centers[0], radii[0])

        assert idx_cl[0] == 0
        assert np.isclose(dist_cl[0], dist_iter)

        # --- Test Case 2: No intersection ---
        centers_no_hit = np.array([[20, 20]], dtype=np.float32)
        dist_cl_no_hit, idx_cl_no_hit = opencl_vectorized_line_circle_intersection(p1s, p2s, centers_no_hit, radii)

        assert idx_cl_no_hit[0] == -1
        assert np.isinf(dist_cl_no_hit[0])

        # --- Test Case 3: Multiple whiskers and multiple circles ---
        p1s_multi = np.array([[0, 0], [10, 10]], dtype=np.float32)
        p2s_multi = np.array([[10, 0], [0, 10]], dtype=np.float32)
        centers_multi = np.array([[5, 0], [5, 10], [20, 20]], dtype=np.float32)
        radii_multi = np.array([2, 2, 2], dtype=np.float32)

        dist_cl_multi, idx_cl_multi = opencl_vectorized_line_circle_intersection(p1s_multi, p2s_multi, centers_multi, radii_multi)

        # Whisker 0 should hit circle 0
        dist_iter_0 = iterative_line_circle_intersection(p1s_multi[0], p2s_multi[0], centers_multi[0], radii_multi[0])
        assert idx_cl_multi[0] == 0
        assert np.isclose(dist_cl_multi[0], dist_iter_0)

        # Whisker 1 should hit circle 1
        dist_iter_1 = iterative_line_circle_intersection(p1s_multi[1], p2s_multi[1], centers_multi[1], radii_multi[1])
        assert idx_cl_multi[1] == 1
        assert np.isclose(dist_cl_multi[1], dist_iter_1)
