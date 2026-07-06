"""Backend-independent layer interfaces for the MobileNetV2 model.

The concrete implementations live in ``backends``. This allows the model to
switch between NumPy, CuPy or custom CUDA implementations without changing
its architecture.

Needed blocks:
- ConvBN
    - Convolution
    - Batch normalization
- InvertedBottleneck
    - Convolution
    - Batch normalization
    - Activation function
    - Residual connection (optional)
- Average pooling
- Fully connected layer
    - Linear layer
    - Activation function
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


Tensor = TypeVar("Tensor")


class Layer(ABC, Generic[Tensor]):
    """Base interface for all backend-independent layers."""

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Calculate the output of the layer."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: Tensor) -> Tensor:
        """Calculate the gradient of the layer with respect to its inputs."""
        raise NotImplementedError
