"""MobileNetV2 architecture adapted to 64x64 GTSRB images."""

from dataclasses import dataclass
from typing import Any, Protocol

from src.blocks import ConvBNReLU6, InvertedBottleneck


MNV2_GTSRB_SPEC = {
    "spec_name": "MobileNetV2-GTSRB",
    "input_size": (64, 64),
    "num_classes": 43,
    "block_specs": [
        ("convbn", 3, 1, 32, None, False),           # 64 -> 64
        ("invertedbottleneck", 3, 1, 16, 1.0, False),
        ("invertedbottleneck", 3, 2, 24, 6.0, False), # 64 -> 32
        ("invertedbottleneck", 3, 1, 24, 6.0, True),
        ("invertedbottleneck", 3, 2, 32, 6.0, False), # 32 -> 16
        ("invertedbottleneck", 3, 1, 32, 6.0, False),
        ("invertedbottleneck", 3, 1, 32, 6.0, True),
        ("invertedbottleneck", 3, 2, 64, 6.0, False), # 16 -> 8
        ("invertedbottleneck", 3, 1, 64, 6.0, False),
        ("invertedbottleneck", 3, 1, 64, 6.0, False),
        ("invertedbottleneck", 3, 1, 64, 6.0, False),
        ("invertedbottleneck", 3, 1, 96, 6.0, False),
        ("invertedbottleneck", 3, 1, 96, 6.0, False),
        ("invertedbottleneck", 3, 1, 96, 6.0, True),
        ("invertedbottleneck", 3, 2, 160, 6.0, False), # 8 -> 4
        ("invertedbottleneck", 3, 1, 160, 6.0, False),
        ("invertedbottleneck", 3, 1, 160, 6.0, False),
        ("invertedbottleneck", 3, 1, 320, 6.0, True),
        ("convbn", 1, 1, 1280, None, False),
    ],
}


class Backend(Protocol):
    convolution: Any
    batch_norm: Any
    activation: Any
    average_pooling: Any
    linear: Any


@dataclass(frozen=True)
class ModelShape:
    channels: int
    height: int
    width: int


class MobileNetV2:
    """Backend-independent MobileNetV2 assembled from layer factories."""

    def __init__(
        self,
        backend: Backend,
        spec: dict[str, Any] = MNV2_GTSRB_SPEC,
        in_channels: int = 3,
    ) -> None:
        self.backend = backend
        self.spec = spec
        self.num_classes = int(spec["num_classes"])
        self.features: list[Any] = []

        input_height, input_width = spec["input_size"]
        shape = ModelShape(in_channels, input_height, input_width)

        for block_spec in spec["block_specs"]:
            (
                block_name,
                kernel_size,
                stride,
                out_channels,
                expand_ratio,
                _is_output,
            ) = block_spec

            if block_name == "convbn":
                block = self._make_conv_bn(
                    in_channels=shape.channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=1,
                    activate=True,
                )
            elif block_name == "invertedbottleneck":
                block = self._make_inverted_bottleneck(
                    in_channels=shape.channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_ratio=expand_ratio,
                )
            else:
                raise ValueError(f"unknown block type: {block_name}")

            self.features.append(block)
            shape = ModelShape(
                channels=out_channels,
                height=self._output_size(shape.height, kernel_size, stride),
                width=self._output_size(shape.width, kernel_size, stride),
            )

        if shape.height != shape.width:
            raise ValueError("global pooling currently requires square features")

        self.output_shape = shape
        self.pooling = backend.average_pooling(
            kernel_size=shape.height,
            stride=shape.height,
        )
        self.classifier = backend.linear(
            in_features=shape.channels,
            out_features=self.num_classes,
            bias=True,
        )

    @staticmethod
    def _output_size(size: int, kernel_size: int, stride: int) -> int:
        padding = kernel_size // 2
        return (size + 2 * padding - kernel_size) // stride + 1

    def _make_conv_bn(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        activate: bool,
    ) -> ConvBNReLU6:
        convolution = self.backend.convolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        batch_norm = self.backend.batch_norm(out_channels)
        activation = self.backend.activation() if activate else None
        return ConvBNReLU6(convolution, batch_norm, activation)

    def _make_inverted_bottleneck(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
    ) -> InvertedBottleneck:
        expanded_channels = int(round(in_channels * expand_ratio))
        expand = None

        if expanded_channels != in_channels:
            expand = self._make_conv_bn(
                in_channels=in_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                activate=True,
            )

        depthwise = self._make_conv_bn(
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=expanded_channels,
            activate=True,
        )
        project = self._make_conv_bn(
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            activate=False,
        )
        return InvertedBottleneck(
            expand=expand,
            depthwise=depthwise,
            project=project,
            use_residual=stride == 1 and in_channels == out_channels,
        )

    def __call__(self, inputs: Any, *, training: bool = False) -> Any:
        return self.forward(inputs, training=training)

    def forward(self, inputs: Any, *, training: bool = False) -> Any:
        outputs = inputs

        for block in self.features:
            outputs = block(outputs, training=training)

        outputs = self.pooling(outputs)
        outputs = outputs.reshape(outputs.shape[0], -1)
        return self.classifier(outputs)
