
import cupy as cp

from src.layers.batch_norm import BatchNormalization


class CuPyBatchNorm(BatchNormalization[cp.ndarray]):
    """CuPy implementation of the batch normalization layer."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        super().__init__(num_features, eps, momentum)

        self.inputs: cp.ndarray | None = None
        self.batch_mean: cp.ndarray | None = None
        self.batch_var: cp.ndarray | None = None
        self.normalized: cp.ndarray | None = None
        self.training = False
        self.gamma = cp.ones(num_features, dtype=cp.float32)
        self.beta = cp.zeros(num_features, dtype=cp.float32)
        self.grad_gamma = cp.zeros(num_features, dtype=cp.float32)
        self.grad_beta = cp.zeros(num_features, dtype=cp.float32)
        self.running_mean = cp.zeros(num_features, dtype=cp.float32)
        self.running_var = cp.ones(num_features, dtype=cp.float32)


    def forward(self, inputs: cp.ndarray, *, training: bool = False) -> cp.ndarray:
        """Apply batch normalization to an NCHW input tensor."""
        if not isinstance(inputs, cp.ndarray):
            raise TypeError("inputs must be a CuPy array")

        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (N, C, H, W)")

        channels = inputs.shape[1]

        if channels != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} channels but got {channels}"
            )

        self.training = training
        if training:
            mean = inputs.mean(axis=(0, 2, 3))
            var = inputs.var(axis=(0, 2, 3))

            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        self.inputs = inputs
        self.batch_mean = mean
        self.batch_var = var

        normalized = (inputs - mean[None, :, None, None]) / cp.sqrt(
            var[None, :, None, None] + self.eps
        )
        self.normalized = normalized

        return (
            self.gamma[None, :, None, None] * normalized
            + self.beta[None, :, None, None]
        )
    
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Calculate the gradient of the batch normalization with respect to its inputs."""
        if not isinstance(grad, cp.ndarray):
            raise TypeError("grad must be a CuPy array")
        if self.inputs is None or self.batch_mean is None or self.batch_var is None:
            raise RuntimeError("forward must be called before backward")

        if grad.ndim != 4:
            raise ValueError("grad must have shape (N, C, H, W)")

        channels = grad.shape[1]

        if channels != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} channels but got {channels}"
            )

        N, C, H, W = grad.shape
        sample_count = N * H * W
        mean = self.batch_mean
        var = self.batch_var
        normalized = self.normalized

        std_inv = 1.0 / cp.sqrt(var + self.eps)
        self.grad_gamma = cp.sum(grad * normalized, axis=(0, 2, 3))
        self.grad_beta = cp.sum(grad, axis=(0, 2, 3))
        grad_normalized = grad * self.gamma[None, :, None, None]

        if not self.training:
            return grad_normalized * std_inv[None, :, None, None]

        grad_input = (
            (1.0 / sample_count)
            * std_inv[None, :, None, None]
            * (
                sample_count * grad_normalized
                - cp.sum(grad_normalized, axis=(0, 2, 3))[None, :, None, None]
                - (self.inputs - mean[None, :, None, None])
                * std_inv[None, :, None, None] ** 2
                * cp.sum(
                    grad_normalized
                    * (self.inputs - mean[None, :, None, None]),
                    axis=(0, 2, 3),
                )[None, :, None, None]
            )
        )

        return grad_input
