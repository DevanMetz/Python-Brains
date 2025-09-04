import numpy as np
from mlp import MLP

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

if OPENCL_AVAILABLE:
    class MLPOpenCL(MLP):
        """
        An MLP implementation that uses OpenCL to accelerate the forward pass.
        Falls back to the NumPy-based MLP if PyOpenCL is not available.
        """
        _opencl_kernel_code = """
        __kernel void forward_pass(__global const float *input,
                                   __global const float *weights,
                                   __global const float *biases,
                                   __global float *output,
                                   int input_size,
                                   int output_size)
        {
            int i = get_global_id(0); // Index for the output neuron
            if (i < output_size)
            {
                float sum = 0.0f;
                for (int j = 0; j < input_size; ++j)
                {
                    sum += input[j] * weights[j * output_size + i];
                }
                sum += biases[i];
                output[i] = tanh(sum);
            }
        }
        """

        def __init__(self, layer_sizes, ctx=None, verbose=False):
            super().__init__(layer_sizes)

            if ctx is None:
                try:
                    self.ctx = cl.create_some_context(interactive=False)
                    if verbose:
                        print(f"Using OpenCL device: {self.ctx.devices[0].name}")
                except cl.LogicError:
                    if verbose:
                        print("No OpenCL context found. Falling back to NumPy.")
                    self._disable_opencl()
                    return
            else:
                self.ctx = ctx

            self.queue = cl.CommandQueue(self.ctx)

            # Transfer weights and biases to GPU
            self.weights_gpu = [cl_array.to_device(self.queue, w.astype(np.float32)) for w in self.weights]
            self.biases_gpu = [cl_array.to_device(self.queue, b.astype(np.float32)) for b in self.biases]

            # Build the OpenCL kernel
            self.prg = cl.Program(self.ctx, self._opencl_kernel_code).build()

        def _disable_opencl(self):
            """Disables OpenCL functionality for this instance."""
            self.ctx = None
            self.queue = None
            self.weights_gpu = None
            self.biases_gpu = None
            self.prg = None

        def forward(self, inputs):
            """Performs a forward pass using OpenCL if available."""
            if not self.ctx: # Fallback to numpy if OpenCL is disabled
                return super().forward(inputs)

            if not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs, dtype=np.float32)
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)

            current_layer_output_gpu = cl_array.to_device(self.queue, inputs.astype(np.float32))
            activations_gpu = [current_layer_output_gpu]

            for i in range(len(self.weights_gpu)):
                input_size = self.layer_sizes[i]
                output_size = self.layer_sizes[i+1]

                output_gpu = cl_array.empty(self.queue, (1, output_size), dtype=np.float32)

                self.prg.forward_pass(self.queue,
                                      (output_size,),
                                      None,
                                      current_layer_output_gpu.data,
                                      self.weights_gpu[i].data,
                                      self.biases_gpu[i].data,
                                      output_gpu.data,
                                      np.int32(input_size),
                                      np.int32(output_size))

                current_layer_output_gpu = output_gpu
                activations_gpu.append(current_layer_output_gpu)

            # Transfer final result and activations back to CPU
            final_output = current_layer_output_gpu.get()
            activations = [act.get() for act in activations_gpu]

            return final_output, activations

        def mutate(self, mutation_rate=0.01, mutation_amount=0.1):
            """Mutates the MLP and updates the GPU buffers."""
            super().mutate(mutation_rate, mutation_amount)
            if self.ctx:
                # After mutating on CPU, re-upload to GPU
                self.weights_gpu = [cl_array.to_device(self.queue, w.astype(np.float32)) for w in self.weights]
                self.biases_gpu = [cl_array.to_device(self.queue, b.astype(np.float32)) for b in self.biases]

        def clone(self):
            """Creates a deep copy of the MLPOpenCL instance."""
            if not self.ctx:
                return super().clone()

            # Create a new instance of the same class, passing the context
            cloned_mlp = self.__class__(self.layer_sizes, ctx=self.ctx)

            # Copy the weights and biases from CPU arrays
            cloned_mlp.weights = [w.copy() for w in self.weights]
            cloned_mlp.biases = [b.copy() for b in self.biases]

            # Transfer the copied weights and biases to the new instance's GPU buffers
            cloned_mlp.weights_gpu = [cl_array.to_device(cloned_mlp.queue, w.astype(np.float32)) for w in cloned_mlp.weights]
            cloned_mlp.biases_gpu = [cl_array.to_device(cloned_mlp.queue, b.astype(np.float32)) for b in cloned_mlp.biases]

            return cloned_mlp

else:
    # Fallback to the standard MLP if PyOpenCL is not installed
    class MLPOpenCL(MLP):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if kwargs.get('verbose', False):
                print("Warning: PyOpenCL not found. Falling back to NumPy-based MLP.")
