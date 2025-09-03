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
        This test uses the new optimized path with cached buffers.
        """
        from mlp_opencl import context
        import pyopencl as cl

        layer_sizes = [10, 20, 5]
        numpy_mlp = MLP(layer_sizes)
        opencl_mlp = MLPOpenCL(layer_sizes)
        opencl_mlp.weights = [w.copy() for w in numpy_mlp.weights]
        opencl_mlp.biases = [b.copy() for b in numpy_mlp.biases]

        inputs = np.random.rand(1, layer_sizes[0]).astype(np.float32)

        # Manually create the buffers as the trainer would
        weights_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w.astype(np.float32)) for w in opencl_mlp.weights]
        biases_bufs = [cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.astype(np.float32)) for b in opencl_mlp.biases]
        max_layer_size = max(opencl_mlp.layer_sizes)
        activations_buf_size = max_layer_size * 2 * np.dtype(np.float32).itemsize
        activations_buf = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=activations_buf_size)

        cached_buffers = {
            'weights': weights_bufs,
            'biases': biases_bufs,
            'activations_buf': activations_buf,
            'max_layer_size': max_layer_size
        }

        # Get the output from both MLPs
        numpy_output = numpy_mlp.forward(inputs)
        opencl_output = opencl_mlp.forward(inputs, cached_buffers=cached_buffers)

        # Release buffers
        for buf in weights_bufs: buf.release()
        for buf in biases_bufs: buf.release()
        activations_buf.release()

        assert np.allclose(numpy_output, opencl_output, atol=1e-6), "Optimized OpenCL MLP output does not match NumPy MLP output."

    def test_fallback_without_cache(self):
        """
        Tests that the OpenCL MLP correctly falls back to the NumPy implementation
        when cached_buffers are not provided.
        """
        layer_sizes = [8, 4]
        opencl_mlp = MLPOpenCL(layer_sizes)
        inputs = np.random.rand(1, layer_sizes[0]).astype(np.float32)

        # Get the output from the standard numpy implementation
        numpy_output = opencl_mlp.forward(inputs)
        # Get the output from the OpenCL implementation without providing a cache
        opencl_output = opencl_mlp.forward(inputs, cached_buffers=None)

        assert np.allclose(numpy_output, opencl_output, atol=1e-6), "Fallback OpenCL output does not match NumPy MLP output."

    def test_cloned_instance_forward_pass(self):
        """
        Tests that a cloned MLPOpenCL instance can still perform a forward pass.
        """
        layer_sizes = [5, 5]
        opencl_mlp = MLPOpenCL(layer_sizes)
        inputs = np.random.rand(1, layer_sizes[0]).astype(np.float32)

        cloned_mlp = opencl_mlp.clone()

        assert isinstance(cloned_mlp, MLPOpenCL)

        # Calling forward without a cache should work and use the numpy fallback.
        try:
            cloned_mlp.forward(inputs, cached_buffers=None)
        except Exception as e:
            pytest.fail(f"Cloned MLPOpenCL instance failed forward pass with exception: {e}")


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
