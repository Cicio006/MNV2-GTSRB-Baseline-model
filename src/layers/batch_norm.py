from abc import ABC, abstractmethod

from .base import Layer, Tensor


class BatchNormalization(Layer[Tensor], ABC):
    """Abstract interface for batch normalization."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ) -> None:
        if num_features <= 0:
            raise ValueError("num_features must be greater than zero")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be between zero and one")

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

    def __call__(self, inputs: Tensor, *, training: bool = False) -> Tensor:
        return self.forward(inputs, training=training)

    @abstractmethod
    def forward(self, inputs: Tensor, *, training: bool = False) -> Tensor:
        """Apply the backend-specific batch normalization to the input tensor."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Calculate the gradient of the batch normalization with respect to its inputs."""
        raise NotImplementedError
