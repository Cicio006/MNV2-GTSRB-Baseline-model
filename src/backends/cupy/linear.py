import cupy as cp

from src.layers.linear import Linear


class CuPyLinear(Linear[cp.ndarray]):
    """CuPy reference implementation of a fully connected layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias)

        scale = cp.sqrt(2.0 / in_features)
        self.weight = (
            cp.random.randn(out_features, in_features).astype(cp.float32)
            * scale
        )
        self.bias_value = (
            cp.zeros(out_features, dtype=cp.float32)
            if bias
            else None
        )
        self.inputs: cp.ndarray | None = None
        self.grad_weight = cp.zeros_like(self.weight)
        self.grad_bias = (
            cp.zeros_like(self.bias_value)
            if self.bias_value is not None
            else None
        )

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        """Apply ``inputs @ weight.T + bias`` to the final input dimension."""
        if not isinstance(inputs, cp.ndarray):
            raise TypeError("inputs must be a CuPy array")
        if inputs.ndim == 0:
            raise ValueError("inputs must have at least one dimension")
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected {self.in_features} input features, "
                f"got {inputs.shape[-1]}"
            )

        self.inputs = inputs
        output = cp.matmul(inputs, self.weight.T)

        if self.bias_value is not None:
            output = output + self.bias_value

        return output

    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Calculate the gradient of the linear transformation with respect to its inputs."""
        if not isinstance(grad, cp.ndarray):
            raise TypeError("grad must be a CuPy array")
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward")
        if grad.ndim == 0:
            raise ValueError("grad must have at least one dimension")
        if grad.shape[-1] != self.out_features:
            raise ValueError(
                f"expected {self.out_features} output features, "
                f"got {grad.shape[-1]}"
            )

        inputs_2d = self.inputs.reshape(-1, self.in_features)
        grad_2d = grad.reshape(-1, self.out_features)
        self.grad_weight = cp.matmul(grad_2d.T, inputs_2d)

        if self.bias_value is not None:
            self.grad_bias = grad_2d.sum(axis=0)

        return cp.matmul(grad, self.weight)
