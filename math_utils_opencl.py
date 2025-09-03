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
            float2 f = p1 - centers[j];
            float a = line_length_sq;
            float b = 2.0f * dot(f, d);
            float c = dot(f, f) - radii[j] * radii[j];
            float discriminant = b*b - 4.0f*a*c;

            if (discriminant >= 0.0f) {
                float sqrt_disc = sqrt(discriminant);
                float t1 = (-b - sqrt_disc) / (2.0f * a);
                float t2 = (-b + sqrt_disc) / (2.0f * a);

                if (t1 >= 0.0f && t1 <= 1.0f && t1 < closest_t) {
                    closest_t = t1;
                    closest_idx = j;
                }
                if (t2 >= 0.0f && t2 <= 1.0f && t2 < closest_t) {
                    closest_t = t2;
                    closest_idx = j;
                }
            }
        }
        if (closest_idx != -1) {
            closest_dist = closest_t * line_length;
        }
    }

    // --- Part 2: Wall Intersection (Conditional) ---
    float wall_dist = INFINITY;
    if (detect_walls != 0) {
        float dist = length(d);
        int steps = (int)dist;
        if (steps > 0) {
            float2 step_vec = d / (float)steps;
            for (int j = 1; j <= steps; ++j) {
                float2 current_pos = p1 + step_vec * (float)j;
                int grid_x = (int)(current_pos.x / tile_size);
                int grid_y = (int)(current_pos.y / tile_size);
                int tile_index = grid_y * map_width_tiles + grid_x;

                if (tile_map_data[tile_index] == 1) { // 1 == Tile.WALL
                    wall_dist = length(current_pos - p1);
                    break; // Found first wall, can stop checking
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

def opencl_unified_perception(p1s_np, p2s_np, centers_np, radii_np, tile_map_buf, map_width_tiles, tile_size, detect_circles=True, detect_walls=True):
    """
    Python wrapper for the unified OpenCL perception kernel.
    """
    if not unified_perception_kernel or not OPENCL_AVAILABLE:
        raise RuntimeError("OpenCL is not available or the unified kernel has not been compiled.")

    # If nothing is being detected, return empty results immediately.
    if not detect_circles and not detect_walls:
        return np.full(p1s_np.shape[0], np.inf, dtype=np.float32), np.full(p1s_np.shape[0], -1, dtype=np.int32)

    num_whiskers = p1s_np.shape[0]
    num_circles = centers_np.shape[0]

    # --- Create Buffers ---
    p1s_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p1s_np.astype(np.float32))
    p2s_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p2s_np.astype(np.float32))

    # Handle case where there are no circles to avoid creating a zero-size buffer
    if num_circles > 0:
        centers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=centers_np.astype(np.float32))
        radii_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii_np.astype(np.float32))
    else:
        # Create dummy buffers of size 1, they won't be read by the kernel since num_circles is 0
        centers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY, 1)
        radii_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY, 1)

    out_distances_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=num_whiskers * np.dtype(np.float32).itemsize)
    out_indices_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=num_whiskers * np.dtype(np.int32).itemsize)

    # --- Execute Kernel ---
    unified_perception_kernel(
        queue, (num_whiskers,), None,
        p1s_buf, p2s_buf,
        centers_buf, radii_buf, np.int32(num_circles),
        tile_map_buf, np.int32(map_width_tiles), np.int32(tile_size),
        np.int32(detect_circles), np.int32(detect_walls),
        out_distances_buf, out_indices_buf
    ).wait()

    # --- Read Results ---
    out_distances_np = np.empty(num_whiskers, dtype=np.float32)
    out_indices_np = np.empty(num_whiskers, dtype=np.int32)
    cl.enqueue_copy(queue, out_distances_np, out_distances_buf)
    cl.enqueue_copy(queue, out_indices_np, out_indices_buf)

    # --- Release Buffers ---
    p1s_buf.release()
    p2s_buf.release()
    centers_buf.release()
    radii_buf.release()
    out_distances_buf.release()
    out_indices_buf.release()

    return out_distances_np, out_indices_np
