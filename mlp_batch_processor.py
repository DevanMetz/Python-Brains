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
        #pragma OPENCL EXTENSION cl_khr_fp16 : enable

        __kernel void forward_layer_batch(__global const half *input_batch,
                                          __global const half *weights_batch,
                                          __global const half *biases_batch,
                                          __global half *output_batch,
                                          int input_size,
                                          int output_size)
        {
            int unit_idx = get_global_id(0);
            int neuron_idx = get_global_id(1);

            half sum = 0.0h;
            for (int j = 0; j < input_size; ++j) {
                half input_val = input_batch[unit_idx * input_size + j];
                long weight_idx = (long)unit_idx * (input_size * output_size) + (long)j * output_size + neuron_idx;
                half weight_val = weights_batch[weight_idx];
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
                    if 'cl_khr_fp16' not in self.ctx.devices[0].extensions:
                        if verbose: print("MLPBatchProcessor: fp16 not supported. GPU acceleration disabled.")
                        self.ctx = None
                        return
                    if verbose:
                        print(f"MLPBatchProcessor using OpenCL device: {self.ctx.devices[0].name}")
                except Exception:
                    if verbose:
                        print("MLPBatchProcessor: No OpenCL context found. GPU acceleration disabled.")
                    return
            else:
                self.ctx = ctx

            self.queue = cl.CommandQueue(self.ctx)

            self.dtype = np.float16

            self.weights_gpu = []
            self.biases_gpu = []
            for i in range(len(mlp_arch) - 1):
                w_shape = (population_size, mlp_arch[i], mlp_arch[i+1])
                self.weights_gpu.append(cl_array.empty(self.queue, w_shape, dtype=self.dtype))
                b_shape = (population_size, mlp_arch[i+1])
                self.biases_gpu.append(cl_array.empty(self.queue, b_shape, dtype=self.dtype))

            self.prg = cl.Program(self.ctx, self._opencl_kernel_code).build()
            self.kernel = self.prg.forward_layer_batch

        def update_brain_on_gpu(self, unit_id: int, mlp: MLP):
            if not self.ctx or unit_id >= self.population_size: return
            for i in range(len(mlp.weights)):
                self.weights_gpu[i][unit_id].set(mlp.weights[i].astype(self.dtype))
                self.biases_gpu[i][unit_id].set(mlp.biases[i].flatten().astype(self.dtype))

        def forward_batch(self, inputs_batch: np.ndarray):
            if not self.kernel: return None

            inputs_gpu = cl_array.to_device(self.queue, inputs_batch.astype(self.dtype))
            current_layer_output_gpu = inputs_gpu

            for i in range(len(self.mlp_arch) - 1):
                input_size = self.mlp_arch[i]
                output_size = self.mlp_arch[i+1]

                output_shape = (self.population_size, output_size)
                output_gpu = cl_array.empty(self.queue, output_shape, dtype=self.dtype)

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

            return current_layer_output_gpu.get()

else:
    class MLPBatchProcessor:
        def __init__(self, *args, **kwargs): pass
        def update_brain_on_gpu(self, *args, **kwargs): pass
        def forward_batch(self, *args, **kwargs): return None
