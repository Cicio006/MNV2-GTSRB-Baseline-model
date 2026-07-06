from typing import Generic

from src.blocks.base import Block
from src.layers import Activation, BatchNormalization, Convolution
from src.layers.base import Tensor


class ConvBNReLU6(Block[Tensor], Generic[Tensor]):
    """Convolution followed by batch normalization and optional activation."""

    def __init__(
        self,
        convolution: Convolution[Tensor],
        batch_norm: BatchNormalization[Tensor],
        activation: Activation[Tensor] | None = None,
    ) -> None:
        if convolution.out_channels != batch_norm.num_features:
            raise ValueError(
                "convolution out_channels must match "
                "batch_norm num_features"
            )

        self.convolution = convolution
        self.batch_norm = batch_norm
        self.activation = activation

    def forward(
        self,
        inputs: Tensor,
        *,
        training: bool = False,
    ) -> Tensor:
        outputs = self.convolution(inputs)
        outputs = self.batch_norm(outputs, training=training)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs
    
    def backward(
        self,
        grad: Tensor,
    ) -> Tensor:
        if self.activation is not None:
            grad = self.activation.backward(grad)
        
        grad = self.batch_norm.backward(grad)
        grad = self.convolution.backward(grad)

        return grad
