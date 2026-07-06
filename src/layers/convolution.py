from abc import ABC, abstractmethod

from .base import Layer, Tensor


class Convolution(Layer[Tensor], ABC):
    """Abstract interface for a two-dimensional convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        if in_channels <= 0:
            raise ValueError("in_channels must be greater than zero")
        if out_channels <= 0:
            raise ValueError("out_channels must be greater than zero")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be greater than zero")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")
        if padding < 0:
            raise ValueError("padding must not be negative")
        if dilation <= 0:
            raise ValueError("dilation must be greater than zero")
        if groups <= 0:
            raise ValueError("groups must be greater than zero")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if not isinstance(bias, bool):
            raise TypeError("bias must be a boolean")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the backend-specific convolution to the input tensor."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Calculate the gradient of the convolution with respect to its inputs."""
        raise NotImplementedError