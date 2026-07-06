from abc import ABC, abstractmethod

from .base import Layer, Tensor


class Linear(Layer[Tensor], ABC):
    """Abstract interface for a fully connected linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be greater than zero")
        if out_features <= 0:
            raise ValueError("out_features must be greater than zero")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean")

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the backend-specific linear transformation to the input tensor."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Calculate the gradient of the linear transformation with respect to its inputs."""
        raise NotImplementedError