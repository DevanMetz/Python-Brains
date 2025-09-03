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
if OPENCL_AVAILABLE:
    try:
        program = cl.Program(context, kernel_code).build()
    except cl.Error as e:
        print(f"ERROR: Failed to compile OpenCL kernel: {e}")
        # Disable OpenCL if compilation fails
        OPENCL_AVAILABLE = False
        program = None

class MLPOpenCL(MLP):
    """
    An MLP that uses OpenCL for GPU-accelerated forward passes.

    Inherits from the base MLP class to reuse the weight/bias initialization
    and the genetic algorithm methods (crossover, mutation). The `forward`
    method is overridden to use the GPU. This class is now picklable as it
    does not hold any non-picklable OpenCL objects as attributes.
    """
    def forward(self, inputs, *, cached_buffers=None):
        """
        Performs a forward pass using OpenCL, with an option for cached buffers.

        Args:
            inputs (np.ndarray): The input vector for the network.
            cached_buffers (dict, optional): A dictionary containing pre-existing
                OpenCL buffer objects for weights and biases. If provided, this
                avoids transferring this data to the GPU on every call.
                Expected format: {'weights': [buf1, buf2,...], 'biases': [buf1, buf2,...]}

        Returns:
            np.ndarray: The output vector from the network.
        """
        if not OPENCL_AVAILABLE or not program:
            return super().forward(inputs)

        try:
            current_input_np = np.array(inputs, dtype=np.float32).flatten()
            temp_buffers = []

            for i in range(len(self.weights)):
                input_size = self.layer_sizes[i]
                output_size = self.layer_sizes[i+1]

                # Create buffer for the current layer's input
                input_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_input_np)
                temp_buffers.append(input_buf)

                # Use cached buffers if available, otherwise create new ones
                if cached_buffers:
                    weights_buf = cached_buffers['weights'][i]
                    biases_buf = cached_buffers['biases'][i]
                else:
                    weights_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.weights[i].astype(np.float32))
                    biases_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.biases[i].astype(np.float32))
                    temp_buffers.extend([weights_buf, biases_buf])

                output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=output_size * np.dtype(np.float32).itemsize)
                temp_buffers.append(output_buf)

                program.forward_layer(
                    queue, (output_size,), None,
                    input_buf, weights_buf, biases_buf, output_buf,
                    np.int32(input_size), np.int32(output_size)
                )

                output_np = np.empty(output_size, dtype=np.float32)
                cl.enqueue_copy(queue, output_np, output_buf).wait()
                current_input_np = output_np

            return current_input_np.reshape(1, -1)

        except cl.Error as e:
            print(f"ERROR: An OpenCL error occurred during the forward pass: {e}. Falling back to NumPy.")
            return super().forward(inputs)
        finally:
            # Ensure all temporarily created buffers are released
            for buf in temp_buffers:
                buf.release()
