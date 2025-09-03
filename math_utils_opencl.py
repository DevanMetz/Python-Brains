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
__kernel void intersection_kernel(
    __global const float2 *p1s,         // Array of whisker start points
    __global const float2 *p2s,         // Array of whisker end points
    __global const float2 *centers,     // Array of circle center points
    __global const float *radii,        // Array of circle radii
    __global float *min_distances,      // Output: minimum intersection distance for each whisker
    __global int *min_indices,          // Output: index of the circle for the minimum intersection
    const int num_circles)
{
    // Get the global ID, which corresponds to the whisker index.
    int i = get_global_id(0);

    float2 p1 = p1s[i];
    float2 p2 = p2s[i];
    float2 d = p2 - p1;

    float line_length_sq = dot(d, d);
    if (line_length_sq == 0.0f) { // Avoid division by zero for zero-length lines
        min_distances[i] = INFINITY;
        min_indices[i] = -1;
        return;
    }
    float line_length = sqrt(line_length_sq);


    float closest_t = INFINITY;
    int closest_idx = -1;

    // Iterate through all circles for the current whisker.
    for (int j = 0; j < num_circles; ++j) {
        float2 f = p1 - centers[j];

        float a = line_length_sq;
        float b = 2.0f * dot(f, d);
        float c = dot(f, f) - radii[j] * radii[j];

        float discriminant = b*b - 4.0f*a*c;

        // Only proceed if there is a real intersection (discriminant >= 0).
        if (discriminant >= 0.0f) {
            float sqrt_disc = sqrt(discriminant);

            // Calculate the two potential solutions for t.
            float t1 = (-b - sqrt_disc) / (2.0f * a);
            float t2 = (-b + sqrt_disc) / (2.0f * a);

            // Check if the intersection points lie on the line segment [0, 1].
            // We are interested in the smallest, non-negative t value.
            if (t1 >= 0.0f && t1 <= 1.0f) {
                if (t1 < closest_t) {
                    closest_t = t1;
                    closest_idx = j;
                }
            }
            // Check t2 only if it could be a closer, valid intersection.
            if (t2 >= 0.0f && t2 <= 1.0f) {
                if (t2 < closest_t) {
                    closest_t = t2;
                    closest_idx = j;
                }
            }
        }
    }

    if (closest_idx != -1) {
        min_distances[i] = closest_t * line_length;
        min_indices[i] = closest_idx;
    } else {
        min_distances[i] = INFINITY;
        min_indices[i] = -1;
    }
}
"""

program = None
intersection_kernel = None
if OPENCL_AVAILABLE:
    try:
        program = cl.Program(context, kernel_code).build()
        intersection_kernel = program.intersection_kernel
    except cl.Error as e:
        print(f"ERROR: Failed to compile OpenCL math kernel: {e}")
        program = None

def opencl_vectorized_line_circle_intersection(p1s_np, p2s_np, centers_np, radii_np):
    """
    Python wrapper for the OpenCL intersection kernel.

    This function handles the data transfer to and from the GPU and executes
    the compiled OpenCL kernel.
    """
    if not intersection_kernel or not OPENCL_AVAILABLE:
        raise RuntimeError("OpenCL is not available or the kernel has not been compiled.")

    num_whiskers = p1s_np.shape[0]
    num_circles = centers_np.shape[0]

    if num_circles == 0:
        return np.full(num_whiskers, np.inf, dtype=np.float32), np.full(num_whiskers, -1, dtype=np.int32)

    # Ensure data is in the correct format (float32 for OpenCL)
    p1s_np = p1s_np.astype(np.float32)
    p2s_np = p2s_np.astype(np.float32)
    centers_np = centers_np.astype(np.float32)
    radii_np = radii_np.astype(np.float32)

    # Create OpenCL buffers on the device and copy data from the host
    p1s_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p1s_np)
    p2s_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=p2s_np)
    centers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=centers_np)
    radii_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=radii_np)

    # Create output buffers on the device
    min_distances_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=num_whiskers * np.dtype(np.float32).itemsize)
    min_indices_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=num_whiskers * np.dtype(np.int32).itemsize)

    # Execute the kernel
    intersection_kernel(
        queue, (num_whiskers,), None, # Global work size is the number of whiskers
        p1s_buf,
        p2s_buf,
        centers_buf,
        radii_buf,
        min_distances_buf,
        min_indices_buf,
        np.int32(num_circles)
    ).wait()

    # Read the results back from the device to the host
    min_distances_np = np.empty(num_whiskers, dtype=np.float32)
    min_indices_np = np.empty(num_whiskers, dtype=np.int32)
    cl.enqueue_copy(queue, min_distances_np, min_distances_buf)
    cl.enqueue_copy(queue, min_indices_np, min_indices_buf)

    # Release buffers
    p1s_buf.release()
    p2s_buf.release()
    centers_buf.release()
    radii_buf.release()
    min_distances_buf.release()
    min_indices_buf.release()

    return min_distances_np, min_indices_np
