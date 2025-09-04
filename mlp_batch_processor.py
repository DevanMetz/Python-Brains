import numpy as np
from mlp import MLP

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

if OPENCL_AVAILABLE:
    class MLPBatchProcessor:
        """
        Manages a population of MLPs on the GPU for efficient batch processing.
        """
        _opencl_kernel_code = """
        __kernel void forward_layer_batch(__global const float *input_batch,      // Shape: (pop_size, input_size)
                                          __global const float *weights_batch,    // Shape: (pop_size, input_size, output_size)
                                          __global const float *biases_batch,     // Shape: (pop_size, 1, output_size)
                                          __global float *output_batch,     // Shape: (pop_size, output_size)
                                          int input_size,
                                          int output_size)
        {
            int unit_idx = get_global_id(0);
            int neuron_idx = get_global_id(1);

            float sum = 0.0f;
            for (int j = 0; j < input_size; ++j) {
                float input_val = input_batch[unit_idx * input_size + j];
                float weight_val = weights_batch[unit_idx * (input_size * output_size) + j * output_size + neuron_idx];
                sum += input_val * weight_val;
            }

            sum += biases_batch[unit_idx * output_size + neuron_idx];

            output_batch[unit_idx * output_size + neuron_idx] = tanh(sum);
        }
        """

        def __init__(self, population_size, mlp_arch, ctx=None, verbose=False):
            self.population_size = population_size
            self.mlp_arch = mlp_arch
            self.ctx = None
            self.queue = None
            self.kernel = None

            if ctx is None:
                try:
                    self.ctx = cl.create_some_context(interactive=False)
                    if verbose:
                        print(f"MLPBatchProcessor using OpenCL device: {self.ctx.devices[0].name}")
                except Exception:
                    if verbose:
                        print("MLPBatchProcessor: No OpenCL context found. GPU acceleration disabled.")
                    return
            else:
                self.ctx = ctx

            self.queue = cl.CommandQueue(self.ctx)

            # Pre-allocate large buffers on the GPU for the entire population
            self.weights_gpu = []
            self.biases_gpu = []
            for i in range(len(mlp_arch) - 1):
                input_size = mlp_arch[i]
                output_size = mlp_arch[i+1]

                w_shape = (population_size, input_size, output_size)
                w_buffer = cl_array.empty(self.queue, w_shape, dtype=np.float32)
                self.weights_gpu.append(w_buffer)

                b_shape = (population_size, output_size) # Simplified for easier indexing
                b_buffer = cl_array.empty(self.queue, b_shape, dtype=np.float32)
                self.biases_gpu.append(b_buffer)

            # Build the kernel
            self.prg = cl.Program(self.ctx, self._opencl_kernel_code).build()
            self.kernel = self.prg.forward_layer_batch

        def update_brain_on_gpu(self, unit_id: int, mlp: MLP):
            """
            Updates the weights and biases for a single unit on the GPU.
            """
            if not self.ctx or unit_id >= self.population_size:
                return

            for i in range(len(mlp.weights)):
                w_cpu = mlp.weights[i].astype(np.float32)
                self.weights_gpu[i][unit_id].set(w_cpu)

                # Reshape bias to match GPU buffer
                b_cpu = mlp.biases[i].flatten().astype(np.float32)
                self.biases_gpu[i][unit_id].set(b_cpu)

        def forward_batch(self, inputs_batch: np.ndarray):
            """
            Performs a forward pass for the entire batch of inputs.
            """
            if not self.kernel:
                return None

            # Transfer inputs to GPU
            inputs_gpu = cl_array.to_device(self.queue, inputs_batch.astype(np.float32))

            current_layer_output_gpu = inputs_gpu

            for i in range(len(self.mlp_arch) - 1):
                input_size = self.mlp_arch[i]
                output_size = self.mlp_arch[i+1]

                output_shape = (self.population_size, output_size)
                output_gpu = cl_array.empty(self.queue, output_shape, dtype=np.float32)

                self.kernel(self.queue,
                            (self.population_size, output_size),
                            None,
                            current_layer_output_gpu.data,
                            self.weights_gpu[i].data,
                            self.biases_gpu[i].data,
                            output_gpu.data,
                            np.int32(input_size),
                            np.int32(output_size))

                current_layer_output_gpu = output_gpu

            final_output = current_layer_output_gpu.get()
            return final_output

else:
    # If OpenCL is not available, create a dummy class
    class MLPBatchProcessor:
        """A dummy class to use when OpenCL is not available."""
        def __init__(self, *args, **kwargs):
            pass
        def update_brain_on_gpu(self, *args, **kwargs):
            pass
        def forward_batch(self, *args, **kwargs):
            return None
