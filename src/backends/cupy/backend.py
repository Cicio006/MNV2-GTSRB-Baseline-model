from .activation import CuPyReLU6
from .avg_pooling import CuPyAvgPooling
from .batch_norm import CuPyBatchNorm
from .convolution import CuPyConvolution
from .linear import CuPyLinear


class CuPyBackend:
    """Layer factory used to assemble a CuPy MobileNetV2."""

    convolution = CuPyConvolution
    batch_norm = CuPyBatchNorm
    activation = CuPyReLU6
    average_pooling = CuPyAvgPooling
    linear = CuPyLinear
