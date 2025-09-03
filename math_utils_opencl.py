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
    __global const int *whisker_parent_indices,

    // Circle object data
    __global const float2 *centers,
    __global const float *radii,
    const int num_circles,

    // Wall data
    __global const int *tile_map_data,
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

    float closest_dist = INFINITY;
    int closest_idx = -1;

    // --- Part 1: Circle Intersection (Conditional) ---
    if (detect_circles != 0 && line_length_sq > 0.0f) {
        float line_length = sqrt(line_length_sq);
        float closest_t = INFINITY;

        for (int j = 0; j < num_circles; ++j) {
            // Skip self-intersection check: if the whisker's parent index in the
            // circle array is the same as the circle we are checking, skip it.
            if (whisker_parent_indices[i] == j) {
                continue;
            }

            float2 f = p1 - centers[j];
            float a = line_length_sq;
            float b = 2.0f * dot(f, d);
            float c = dot(f, f) - radii[j] * radii[j];
            float discriminant = b*b - 4.0f*a*c;

            if (discriminant >= 0.0f) {
                float sqrt_disc = sqrt(discriminant);
                float t1 = (-b - sqrt_disc) / (2.0f * a);

                // This logic intentionally only checks t1 and ignores t2, to match
                // the behavior of the original, latent bug in the CPU-based
                // iterative_line_circle_intersection function. This ensures
                // "bug-for-bug" compatibility with the original simulation behavior.
                if (t1 >= 0.0f && t1 <= 1.0f) {
                    if (t1 < closest_t) {
                        closest_t = t1;
                        closest_idx = j;
                    }
                }
            }
        }
        if (closest_idx != -1) {
            closest_dist = closest_t * line_length;
        }
    }

    // --- Part 2: Wall Intersection (Ray-Marching) ---
    // This is a simple and robust algorithm that is less efficient than DDA
    // but is more stable and less prone to floating point issues.
    float wall_dist = INFINITY;
    if (detect_walls != 0 && line_length_sq > 0.0f) {
        float line_length = sqrt(line_length_sq);
        int steps = (int)line_length;
        if (steps > 0) {
            float2 dir = d / line_length; // Normalized direction vector
            for (int j = 1; j <= steps; ++j) {
                float2 current_pos = p1 + dir * (float)j;
                int grid_x = (int)(current_pos.x / tile_size);
                int grid_y = (int)(current_pos.y / tile_size);

                // Boundary checks for the map
                if (grid_x >= 0 && grid_x < map_width_tiles && grid_y >= 0) {
                    // A full bounds check would need map_height_tiles, but this is a safe guard.
                    int tile_index = grid_y * map_width_tiles + grid_x;
                    if (tile_map_data[tile_index] == 1) { // 1 == Tile.WALL
                        wall_dist = (float)j; // The distance is the number of steps taken.
                        break;
                    }
                } else {
                    break; // Stop if we go out of map bounds
                }
            }
        }
    }

    // --- Part 3: Combine Results ---
    if (wall_dist < closest_dist) {
        out_distances[i] = wall_dist;
        out_indices[i] = -2; // Special index for wall
    } else {
        out_distances[i] = closest_dist;
        out_indices[i] = closest_idx;
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
    p1s_buf, p2s_buf, centers_buf, radii_buf, out_distances_buf, out_indices_buf,
    whisker_parent_indices_buf,
    num_whiskers, num_circles,
    tile_map_buf, map_width_tiles, tile_size,
    detect_circles=True, detect_walls=True
):
    """
    Executes the unified OpenCL perception kernel with pre-allocated buffers.

    This function is designed for high-performance scenarios. It assumes that
    all OpenCL buffers have been created and that the necessary data has
    already been copied to the device. It only enqueues the kernel execution
    and does not handle buffer creation, data transfer, or synchronization.

    Args:
        queue (cl.CommandQueue): The OpenCL command queue.
        p1s_buf (cl.Buffer): Buffer for whisker start points.
        p2s_buf (cl.Buffer): Buffer for whisker end points.
        centers_buf (cl.Buffer): Buffer for circle center points.
        radii_buf (cl.Buffer): Buffer for circle radii.
        out_distances_buf (cl.Buffer): Output buffer for intersection distances.
        out_indices_buf (cl.Buffer): Output buffer for intersection indices.
        num_whiskers (int): The total number of whiskers to process.
        num_circles (int): The total number of circles to check against.
        tile_map_buf (cl.Buffer): Buffer for the static wall tile map.
        map_width_tiles (int): The width of the map in tiles.
        tile_size (int): The size of each tile in pixels.
        detect_circles (bool): Flag to enable/disable circle detection.
        detect_walls (bool): Flag to enable/disable wall detection.

    Returns:
        cl.Event: The event object associated with the kernel execution.
    """
    if not unified_perception_kernel or not OPENCL_AVAILABLE:
        raise RuntimeError("OpenCL is not available or the unified kernel has not been compiled.")

    # If nothing is being detected, we don't need to launch the kernel.
    # The caller is responsible for handling the empty results.
    if not detect_circles and not detect_walls:
        return None

    # --- Execute Kernel ---
    # The caller is responsible for waiting on the returned event.
    return unified_perception_kernel(
        queue, (num_whiskers,), None,
        p1s_buf, p2s_buf, whisker_parent_indices_buf,
        centers_buf, radii_buf, np.int32(num_circles),
        tile_map_buf, np.int32(map_width_tiles), np.int32(tile_size),
        np.int32(detect_circles), np.int32(detect_walls),
        out_distances_buf, out_indices_buf
    )
