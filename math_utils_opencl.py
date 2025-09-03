"""
Provides OpenCL-based mathematical utility functions for geometry calculations.

This module is a replacement for the CuPy-based vectorized functions, offering
a hardware-agnostic way to perform the same calculations on any GPU that
supports OpenCL.
"""
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# Reuse the context and queue from the MLP OpenCL implementation to avoid
# re-initialization and ensure a single OpenCL context per process.
# This is a cleaner approach for a larger application.
from mlp_opencl import context, queue, OPENCL_AVAILABLE

# --- OpenCL Kernel for Vectorized Line-Circle Intersection ---
# This kernel calculates intersections between N line segments (whiskers) and M circles (objects).
# Each instance of the kernel is responsible for one whisker and checks it against all circles.
kernel_code = """
__kernel void unified_perception_kernel(
    // Whisker data
    __global const float2 *p1s,
    __global const float2 *p2s,

    // World data (replaces separate wall/circle data)
    __global const int *world_grid_data,
    const int map_width_tiles,
    const int tile_size,

    // --- Control Flags ---
    const int detect_circles,
    const int detect_walls,

    // Output
    __global float *out_distances,
    __global int *out_indices
) {
    int i = get_global_id(0); // Whisker index
    float2 p1 = p1s[i];
    float2 p2 = p2s[i];
    float2 d = p2 - p1;
    float line_length_sq = dot(d, d);

    // The kernel is now much simpler. It only performs ray-marching against a unified grid.
    // The output distance is written to out_distances, and the detected object type
    // is written to out_indices.

    // --- Unified Ray-Marching on a Grid ---
    out_distances[i] = INFINITY;
    out_indices[i] = 0; // 0 == EMPTY

    if (line_length_sq > 0.0f) {
        float line_length = sqrt(line_length_sq);
        int steps = (int)line_length;
        if (steps > 0) {
            float2 dir = d / line_length; // Normalized direction vector
            for (int j = 1; j <= steps; ++j) {
                float2 current_pos = p1 + dir * (float)j;
                int grid_x = (int)(current_pos.x / tile_size);
                int grid_y = (int)(current_pos.y / tile_size);

                if (grid_x >= 0 && grid_x < map_width_tiles && grid_y >= 0) {
                    int tile_index = grid_y * map_width_tiles + grid_x;
                    int object_type = world_grid_data[tile_index];

                    // If we hit anything other than empty space, record it and stop.
                    if (object_type > 0) {
                        out_distances[i] = (float)j;
                        out_indices[i] = object_type;
                        break;
                    }
                } else {
                    break; // Stop if we go out of map bounds
                }
            }
        }
    }
}
"""

program = None
unified_perception_kernel = None
if OPENCL_AVAILABLE:
    try:
        program = cl.Program(context, kernel_code).build()
        unified_perception_kernel = program.unified_perception_kernel
    except cl.Error as e:
        print(f"ERROR: Failed to compile OpenCL math kernel: {e}")
        program = None

def opencl_unified_perception(
    queue,
    p1s_buf, p2s_buf,
    world_grid_buf,
    out_distances_buf, out_indices_buf,
    num_whiskers, map_width_tiles, tile_size
):
    """
    Executes the unified OpenCL perception kernel.

    This version uses a simplified, grid-based detection model where all
    objects (walls, units, etc.) are rasterized onto a single grid. The kernel
    performs a ray-march for each whisker on this grid.

    Args:
        queue (cl.CommandQueue): The OpenCL command queue.
        p1s_buf (cl.Buffer): Buffer for whisker start points.
        p2s_buf (cl.Buffer): Buffer for whisker end points.
        world_grid_buf (cl.Buffer): Buffer for the world grid, containing walls and rasterized objects.
        out_distances_buf (cl.Buffer): Output buffer for intersection distances.
        out_indices_buf (cl.Buffer): Output buffer for detected object types.
        num_whiskers (int): The total number of whiskers to process.
        map_width_tiles (int): The width of the map in tiles.
        tile_size (int): The size of each tile in pixels.

    Returns:
        cl.Event: The event object associated with the kernel execution.
    """
    if not unified_perception_kernel or not OPENCL_AVAILABLE:
        raise RuntimeError("OpenCL is not available or the unified kernel has not been compiled.")

    # Execute the simplified kernel
    return unified_perception_kernel(
        queue, (num_whiskers,), None,
        p1s_buf, p2s_buf,
        world_grid_buf,
        np.int32(map_width_tiles), np.int32(tile_size),
        # Control flags are no longer needed but still exist in the kernel signature.
        # We pass 1 to ensure the logic runs.
        np.int32(1), np.int32(1),
        out_distances_buf, out_indices_buf
    )
