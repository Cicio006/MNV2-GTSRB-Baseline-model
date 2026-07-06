from abc import ABC, abstractmethod

from .base import Layer, Tensor


class Activation(Layer[Tensor], ABC):
    """Abstract interface for activation functions."""

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the backend-specific activation function to the input tensor."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Calculate the gradient of the activation function with respect to its inputs."""
        raise NotImplementedError
