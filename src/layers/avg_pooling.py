from abc import ABC, abstractmethod

from .base import Layer, Tensor


class AveragePooling(Layer[Tensor], ABC):
    """Abstract interface for average pooling."""

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        if kernel_size <= 0:
            raise ValueError("kernel_size must be greater than zero")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")
        if padding < 0:
            raise ValueError("padding must not be negative")
        if padding > kernel_size // 2:
            raise ValueError("padding must not exceed half the kernel size")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the backend-specific average pooling to the input tensor."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Calculate the gradient of the average pooling with respect to its inputs."""
        raise NotImplementedError
