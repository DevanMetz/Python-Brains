import numpy as np
from mlp import MLP

# Try to import cupy and check for a working GPU, but don't fail if it's not available
try:
    import cupy as cp
    # A more robust check for availability: will fail if no driver/device
    if cp.cuda.runtime.getDeviceCount() > 0:
        CUPY_AVAILABLE = True
    else:
        CUPY_AVAILABLE = False
except Exception:
    CUPY_AVAILABLE = False

class MLPCupy(MLP):
    """
    An MLP that uses CuPy for GPU-accelerated forward passes, if available.
    The weights and biases are still stored as NumPy arrays to ensure they can be
    pickled by the multiprocessing module.
    """

    def forward(self, inputs):
        """
        Performs a forward pass through the network, using the GPU if possible.

        Args:
            inputs (np.ndarray): The input vector for the network.

        Returns:
            np.ndarray: The output vector from the network, as a NumPy array.
        """
        if not CUPY_AVAILABLE:
            # Fallback to the parent's (NumPy-based) forward method
            return super().forward(inputs)

        try:
            # --- GPU-accelerated forward pass ---

            # 1. Ensure input is a NumPy array and has the correct dimensions
            if not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs)
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)

            # 2. Move data to the GPU
            inputs_gpu = cp.asarray(inputs)
            weights_gpu = [cp.asarray(w) for w in self.weights]
            biases_gpu = [cp.asarray(b) for b in self.biases]

            # 3. Perform calculations on the GPU
            current_layer_output_gpu = inputs_gpu
            for i in range(len(weights_gpu)):
                z_gpu = cp.dot(current_layer_output_gpu, weights_gpu[i]) + biases_gpu[i]
                current_layer_output_gpu = cp.tanh(z_gpu)

            # 4. Move the final result back to the CPU
            output_cpu = cp.asnumpy(current_layer_output_gpu)

            return output_cpu

        except Exception as e:
            print(f"Warning: CuPy forward pass failed with error: {e}. Falling back to NumPy.")
            # Fallback to the parent's (NumPy-based) forward method in case of any CuPy error
            return super().forward(inputs)
