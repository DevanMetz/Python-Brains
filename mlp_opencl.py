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
kernel_code = """
__kernel void forward_layer(
    __global const float *input_buffer,
    __global const float *weights_buffer,
    __global const float *biases_buffer,
    __global float *output_buffer,
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
            sum += input_buffer[j] * weights_buffer[j * output_size + i];
        }
        // Add the bias for this neuron.
        sum += biases_buffer[i];
        // Apply the tanh activation function and store the result.
        output_buffer[i] = tanh(sum);
    }
}
"""

# Compile the kernel globally if OpenCL is available
forward_layer_kernel = None
if OPENCL_AVAILABLE:
    try:
        program = cl.Program(context, kernel_code).build()
        forward_layer_kernel = program.forward_layer
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
    "ping-pong" strategy with two pre-allocated buffers for all layer activations.
    This avoids the overhead of creating new buffers or copying intermediate
    results back to the CPU for each layer.

    Inherits from the base MLP class to reuse the weight/bias initialization
    and the genetic algorithm methods (crossover, mutation).
    """
    def forward(self, inputs, *, cached_buffers=None):
        """
        Performs a highly optimized forward pass using OpenCL.

        This method executes the entire forward pass on the GPU. It requires a
        `cached_buffers` dictionary containing pre-allocated OpenCL buffers for
        weights, biases, and two intermediate buffers for the ping-ponging.

        Args:
            inputs (np.ndarray): The input vector for the network.
            cached_buffers (dict): A dictionary containing pre-existing OpenCL
                buffer objects. This is not optional for this implementation.
                Expected format: {
                    'weights': [buf1, buf2,...],
                    'biases': [buf1, buf2,...],
                    'intermediate_a': buffer,
                    'intermediate_b': buffer
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
            # The two buffers for ping-ponging intermediate results
            intermediate_buf_a = cached_buffers['intermediate_a']
            intermediate_buf_b = cached_buffers['intermediate_b']

            # 1. Copy initial input data to the first intermediate buffer on the GPU.
            input_np = np.array(inputs, dtype=np.float32).flatten()
            cl.enqueue_copy(queue, intermediate_buf_a, input_np, is_blocking=False)

            # 2. Sequentially execute kernels for each layer, staying on the GPU.
            current_input_buf = intermediate_buf_a
            current_output_buf = intermediate_buf_b

            for i in range(len(self.weights)):
                input_size = self.layer_sizes[i]
                output_size = self.layer_sizes[i+1]
                weights_buf = weights_bufs[i]
                biases_buf = biases_bufs[i]

                # Launch the kernel for the current layer.
                forward_layer_kernel(
                    queue, (output_size,), None,
                    current_input_buf, weights_buf, biases_buf, current_output_buf,
                    np.int32(input_size), np.int32(output_size)
                )

                # Swap buffers for the next layer. The previous output is the new input.
                current_input_buf, current_output_buf = current_output_buf, current_input_buf

            # 3. Copy the final result back from the GPU to the CPU.
            # The final result is located in the last `current_input_buf` (due to the swap).
            output_size = self.layer_sizes[-1]
            output_np = np.empty(output_size, dtype=np.float32)
            cl.enqueue_copy(queue, output_np, current_input_buf).wait()

            return output_np.reshape(1, -1)

        except (cl.Error, KeyError) as e:
            print(f"ERROR: An OpenCL error or a key error occurred during the optimized forward pass: {e}. Falling back to NumPy.")
            return super().forward(inputs)
