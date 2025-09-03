import pytest
import numpy as np

# Conditional import of OpenCL modules to allow tests to be collected even if pyopencl is not installed
try:
    import pygame
    from mlp_opencl import MLPOpenCL, OPENCL_AVAILABLE
    from math_utils_opencl import opencl_unified_perception
    from map import TileMap, Tile
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
        intermediate_buf_size = max_layer_size * np.dtype(np.float32).itemsize
        intermediate_buf_a = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=intermediate_buf_size)
        intermediate_buf_b = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=intermediate_buf_size)


        cached_buffers = {
            'weights': weights_bufs,
            'biases': biases_bufs,
            'intermediate_a': intermediate_buf_a,
            'intermediate_b': intermediate_buf_b
        }

        # Get the output from both MLPs
        numpy_output = numpy_mlp.forward(inputs)
        opencl_output = opencl_mlp.forward(inputs, cached_buffers=cached_buffers)

        # Release buffers
        for buf in weights_bufs: buf.release()
        for buf in biases_bufs: buf.release()
        intermediate_buf_a.release()
        intermediate_buf_b.release()

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
    def test_unified_perception_correctness(self):
        """
        Tests if the unified OpenCL perception kernel produces the same result
        as the original, iterative CPU-based logic. This is the primary
        correctness test for the perception system.
        """
        from math_utils_opencl import context
        import pyopencl as cl

        # --- 1. Setup Test Scenario ---
        # Create a simple tile map (e.g., 20x15) with a vertical wall
        tile_map = TileMap(200, 150, 10) # 20x15 tiles of size 10
        for y in range(5, 10):
            tile_map.set_tile(10, y, Tile.WALL) # Wall at x=100px

        # Create circle objects
        centers = np.array([[50, 75]], dtype=np.float32) # One circle
        radii = np.array([10], dtype=np.float32)

        # Create whiskers from a single point
        p1s = np.array([[20, 75], [20, 75], [20, 0]], dtype=np.float32)
        p2s = np.array([
            [120, 75], # Whisker 0: Should hit the wall at x=100
            [60, 75],  # Whisker 1: Should hit the circle at x=50
            [20, 20]    # Whisker 2: Should hit nothing
        ], dtype=np.float32)

        # --- 2. Get Ground Truth (CPU Fallback Logic) ---
        expected_dists = []
        expected_indices = []
        for i in range(len(p1s)):
            p1 = pygame.Vector2(p1s[i])
            p2 = pygame.Vector2(p2s[i])
            whisker_length = p1.distance_to(p2)

            closest_dist = np.inf
            detected_type_idx = -1

            # CPU Wall check
            dx, dy = p2.x - p1.x, p2.y - p1.y
            steps = int(max(abs(dx), abs(dy)))
            if steps > 0:
                x_inc, y_inc = dx / steps, dy / steps
                for j in range(1, steps + 1):
                    x, y = p1.x + j * x_inc, p1.y + j * y_inc
                    if tile_map.get_tile_at_pixel(x,y) == Tile.WALL:
                        closest_dist = p1.distance_to(pygame.Vector2(x,y))
                        detected_type_idx = -2 # Wall
                        break

            # CPU Object check
            for obj_idx, center in enumerate(centers):
                dist = iterative_line_circle_intersection(p1, p2, pygame.Vector2(center), radii[obj_idx])
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
                    detected_type_idx = obj_idx

            expected_dists.append(closest_dist if closest_dist < whisker_length else np.inf)
            expected_indices.append(detected_type_idx)

        # --- 3. Get Actual Results (GPU Unified Kernel) ---
        map_data_flat = tile_map.tiles.flatten().astype(np.int32)
        tile_map_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=map_data_flat)

        actual_dists, actual_indices = opencl_unified_perception(
            p1s, p2s, centers, radii, tile_map_buf, tile_map.width, tile_map.tile_size
        )

        tile_map_buf.release()

        # --- 4. Compare and Assert ---
        # Whisker 0 (hits wall)
        assert actual_indices[0] == expected_indices[0]
        assert np.isclose(actual_dists[0], expected_dists[0], atol=2.0)

        # Whisker 1 (hits circle)
        assert actual_indices[1] == expected_indices[1]
        assert np.isclose(actual_dists[1], expected_dists[1])

        # Whisker 2 (hits nothing)
        assert actual_indices[2] == expected_indices[2]
        assert np.isinf(actual_dists[2])

    def test_unified_perception_conditional_logic(self):
        """
        Tests that the conditional flags in the unified kernel work correctly.
        """
        from math_utils_opencl import context
        import pyopencl as cl

        # Setup a scenario with both a wall and a circle
        tile_map = TileMap(200, 150, 10)
        tile_map.set_tile(10, 7, Tile.WALL)
        centers = np.array([[50, 75]], dtype=np.float32)
        radii = np.array([10], dtype=np.float32)
        p1s = np.array([[20, 75]], dtype=np.float32)
        p2s = np.array([[120, 75]], dtype=np.float32) # Aims at both

        map_data_flat = tile_map.tiles.flatten().astype(np.int32)
        tile_map_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=map_data_flat)

        # Case 1: Detect only circles
        dists, indices = opencl_unified_perception(
            p1s, p2s, centers, radii, tile_map_buf, tile_map.width, tile_map.tile_size,
            detect_circles=True, detect_walls=False
        )
        assert indices[0] == 0 # Should detect the circle

        # Case 2: Detect only walls
        dists, indices = opencl_unified_perception(
            p1s, p2s, centers, radii, tile_map_buf, tile_map.width, tile_map.tile_size,
            detect_circles=False, detect_walls=True
        )
        assert indices[0] == -2 # Should detect the wall

        # Case 3: Detect nothing
        dists, indices = opencl_unified_perception(
            p1s, p2s, centers, radii, tile_map_buf, tile_map.width, tile_map.tile_size,
            detect_circles=False, detect_walls=False
        )
        assert indices[0] == -1 # Should detect nothing
        assert np.isinf(dists[0])

        tile_map_buf.release()
