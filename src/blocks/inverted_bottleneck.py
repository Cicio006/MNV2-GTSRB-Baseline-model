from typing import Generic

from src.blocks.base import Block
from src.layers.base import Tensor

from .conv_bn import ConvBNReLU6


class InvertedBottleneck(Block[Tensor], Generic[Tensor]):
    """MobileNetV2 expansion, depthwise and linear projection block."""

    def __init__(
        self,
        expand: ConvBNReLU6[Tensor] | None,
        depthwise: ConvBNReLU6[Tensor],
        project: ConvBNReLU6[Tensor],
        *,
        use_residual: bool,
    ) -> None:
        if project.activation is not None:
            raise ValueError(
                "projection must not have an activation function"
            )

        self.expand = expand
        self.depthwise = depthwise
        self.project = project
        self.use_residual = use_residual

    def forward(
        self,
        inputs: Tensor,
        *,
        training: bool = False,
    ) -> Tensor:
        outputs = inputs

        if self.expand is not None:
            outputs = self.expand(outputs, training=training)

        outputs = self.depthwise(outputs, training=training)
        outputs = self.project(outputs, training=training)

        if self.use_residual:
            outputs = outputs + inputs

        return outputs
    
    def backward(
        self,
        grad: Tensor,
    ) -> Tensor:
        residual_grad = grad

        grad = self.project.backward(grad)
        grad = self.depthwise.backward(grad)

        if self.expand is not None:
            grad = self.expand.backward(grad)

        if self.use_residual:
            grad = grad + residual_grad

        return grad
