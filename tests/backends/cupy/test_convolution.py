import pytest


cp = pytest.importorskip("cupy")

from src.backends.cupy.convolution import CuPyConvolution


def test_convolution_output_shape():
    layer = CuPyConvolution(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=2,
        padding=1,
    )

    inputs = cp.random.randn(2, 3, 64, 64).astype(cp.float32)
    output = layer(inputs)

    assert output.shape == (2, 16, 32, 32)
    assert cp.isfinite(output).all()
