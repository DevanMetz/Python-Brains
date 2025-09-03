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

    def forward(self, inputs, *, _weights_gpu=None, _biases_gpu=None):
        """
        Performs a forward pass, using the GPU if possible.

        This method is optimized to accept pre-cached GPU arrays for weights
        and biases, avoiding redundant data transfers.

        Args:
            inputs (np.ndarray): The input vector for the network.
            _weights_gpu (list, optional): A list of CuPy arrays for the weights.
                If provided, these will be used directly. Defaults to None.
            _biases_gpu (list, optional): A list of CuPy arrays for the biases.
                If provided, these will be used directly. Defaults to None.

        Returns:
            np.ndarray: The output vector, as a NumPy array.
        """
        if not CUPY_AVAILABLE:
            return super().forward(inputs)

        try:
            # --- GPU-accelerated forward pass ---
            if not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs)
            if inputs.ndim == 1:
                inputs = inputs.reshape(1, -1)

            # Move inputs to GPU (this is always necessary)
            inputs_gpu = cp.asarray(inputs)

            # Use cached GPU parameters if provided, otherwise transfer them
            weights_gpu = _weights_gpu if _weights_gpu else [cp.asarray(w) for w in self.weights]
            biases_gpu = _biases_gpu if _biases_gpu else [cp.asarray(b) for b in self.biases]

            # Perform calculations on the GPU
            current_layer_output_gpu = inputs_gpu
            for i in range(len(weights_gpu)):
                z_gpu = cp.dot(current_layer_output_gpu, weights_gpu[i]) + biases_gpu[i]
                current_layer_output_gpu = cp.tanh(z_gpu)

            # Move the final result back to the CPU
            return cp.asnumpy(current_layer_output_gpu)

        except Exception as e:
            print(f"Warning: CuPy forward pass failed with error: {e}. Falling back to NumPy.")
            return super().forward(inputs)
