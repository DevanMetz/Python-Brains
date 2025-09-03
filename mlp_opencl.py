"""
Provides an OpenCL-based implementation of a Multilayer Perceptron (MLP).

This module contains the `MLPOpenCL` class, which is designed to be a
drop-in replacement for the NumPy or CuPy MLP implementations. It offloads
the computationally expensive forward pass to the GPU using the OpenCL
standard, making it compatible with a wide range of hardware, including
NVIDIA and AMD GPUs.
"""
import numpy as np
import pyopencl as cl
from mlp import MLP

# --- OpenCL Initialization ---
OPENCL_AVAILABLE = False
context = None
queue = None
program = None

try:
    # Create a context by trying to find any available GPU platform/device.
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    OPENCL_AVAILABLE = True
    print("SUCCESS: PyOpenCL found a compatible GPU and created a context.")
except Exception as e:
    # This will catch errors like "no devices found" or other OpenCL issues.
    print(f"WARNING: PyOpenCL could not initialize. GPU acceleration will be disabled. Error: {e}")
    OPENCL_AVAILABLE = False


# --- OpenCL Kernel for a single layer's forward pass ---
# This C-like code will be compiled and run on the GPU.
# This optimized kernel uses offsets to work with a single, pre-allocated
# buffer for all layer activations, minimizing buffer creation and data transfers.
kernel_code = """
__kernel void forward_layer_optimized(
    __global float *activations_buffer, // A single buffer for all activations
    __global const float *weights_buffer,
    __global const float *biases_buffer,
    const int input_offset,
    const int output_offset,
    const int input_size,
    const int output_size)
{
    // Get the global ID for this instance of the kernel.
    // This corresponds to the index of the output neuron to compute.
    int i = get_global_id(0);

    // Ensure we don't go out of bounds for the output buffer.
    if (i < output_size) {
        float sum = 0.0f;
        // Perform the dot product of the input vector and the i-th column of the weight matrix.
        for (int j = 0; j < input_size; ++j) {
            // Accessing weights in column-major order (standard for NumPy)
            // Input is read from the shared buffer at a specified offset.
            sum += activations_buffer[input_offset + j] * weights_buffer[j * output_size + i];
        }
        // Add the bias for this neuron.
        sum += biases_buffer[i];
        // Apply the tanh activation function and store the result in the shared buffer
        // at a specified offset.
        activations_buffer[output_offset + i] = tanh(sum);
    }
}
"""

# Compile the kernel globally if OpenCL is available
forward_layer_kernel = None
if OPENCL_AVAILABLE:
    try:
        program = cl.Program(context, kernel_code).build()
        forward_layer_kernel = program.forward_layer_optimized
    except cl.Error as e:
        print(f"ERROR: Failed to compile OpenCL kernel: {e}")
        # Disable OpenCL if compilation fails
        OPENCL_AVAILABLE = False
        program = None

class MLPOpenCL(MLP):
    """
    An MLP that uses OpenCL for GPU-accelerated forward passes.

    This implementation is highly optimized to minimize CPU-GPU data transfers.
    It performs the entire multi-layer forward pass on the GPU by using a
    single, large, pre-allocated buffer for all layer activations. This avoids
    the overhead of creating new buffers or copying intermediate results back
    to the CPU for each layer.

    Inherits from the base MLP class to reuse the weight/bias initialization
    and the genetic algorithm methods (crossover, mutation).
    """
    def forward(self, inputs, *, cached_buffers=None):
        """
        Performs a highly optimized forward pass using OpenCL.

        This method executes the entire forward pass on the GPU. It requires a
        `cached_buffers` dictionary containing pre-allocated OpenCL buffers for
        weights, biases, and a large shared buffer for activations.

        Args:
            inputs (np.ndarray): The input vector for the network.
            cached_buffers (dict): A dictionary containing pre-existing OpenCL
                buffer objects. This is not optional for this implementation.
                Expected format: {
                    'weights': [buf1, buf2,...],
                    'biases': [buf1, buf2,...],
                    'activations_buf': single_large_buffer,
                    'max_layer_size': int
                }

        Returns:
            np.ndarray: The output vector from the network.
        """
        if not forward_layer_kernel or not cached_buffers:
            # Fallback to the slower, CPU-based NumPy implementation if the
            # OpenCL kernel isn't ready or if cached buffers aren't provided.
            return super().forward(inputs)

        try:
            weights_bufs = cached_buffers['weights']
            biases_bufs = cached_buffers['biases']
            activations_buf = cached_buffers['activations_buf']
            max_layer_size = cached_buffers['max_layer_size']

            # 1. Copy initial input data to the start of the activations buffer on the GPU.
            # This is the only host-to-device transfer for the entire forward pass.
            # This call is non-blocking; the subsequent kernel launch will wait for it.
            input_np = np.array(inputs, dtype=np.float32).flatten()
            cl.enqueue_copy(queue, activations_buf, input_np, is_blocking=False, device_offset=0)

            # 2. Sequentially execute kernels for each layer, staying on the GPU.
            # We use a "ping-pong" strategy on the offsets within the single activations buffer.
            input_offset = 0
            output_offset = max_layer_size

            for i in range(len(self.weights)):
                input_size = self.layer_sizes[i]
                output_size = self.layer_sizes[i+1]
                weights_buf = weights_bufs[i]
                biases_buf = biases_bufs[i]

                # Launch the kernel for the current layer.
                forward_layer_kernel(
                    queue, (output_size,), None,
                    activations_buf, weights_buf, biases_buf,
                    np.int32(input_offset), np.int32(output_offset),
                    np.int32(input_size), np.int32(output_size)
                )

                # Swap offsets for the next layer. The previous output is the new input.
                input_offset, output_offset = output_offset, input_offset

            # 3. Copy the final result back from the GPU to the CPU.
            # The final result is located at the last `input_offset`.
            # This is the only device-to-host transfer for the entire forward pass.
            output_size = self.layer_sizes[-1]
            output_np = np.empty(output_size, dtype=np.float32)
            cl.enqueue_copy(queue, output_np, activations_buf, device_offset=input_offset).wait()

            return output_np.reshape(1, -1)

        except (cl.Error, KeyError) as e:
            print(f"ERROR: An OpenCL error or a key error occurred during the optimized forward pass: {e}. Falling back to NumPy.")
            return super().forward(inputs)
