import cupy as cp

from src.layers.activation import Activation


class CuPyReLU6(Activation[cp.ndarray]):
    """CuPy implementation of the ReLU6 activation function."""

    def __init__(self) -> None:
        self.inputs: cp.ndarray | None = None

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        """Apply the ReLU6 activation function to an NCHW input tensor."""
        if not isinstance(inputs, cp.ndarray):
            raise TypeError("inputs must be a CuPy array")

        self.inputs = inputs
        return cp.minimum(cp.maximum(inputs, 0), 6)
    
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Calculate the gradient of the ReLU6 activation function with respect to its inputs."""
        if not isinstance(grad, cp.ndarray):
            raise TypeError("grad must be a CuPy array")
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward")

        mask = (self.inputs > 0) & (self.inputs < 6)
        return grad * mask.astype(grad.dtype)
