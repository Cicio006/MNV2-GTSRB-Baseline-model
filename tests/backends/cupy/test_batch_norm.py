import pytest


cp = pytest.importorskip("cupy")

from src.backends.cupy.batch_norm import CuPyBatchNorm


def test_batch_norm_output_shape():
    layer = CuPyBatchNorm(num_features=16)

    inputs = cp.random.randn(2, 16, 64, 64).astype(cp.float32)
    output = layer(inputs)

    assert output.shape == (2, 16, 64, 64)
    assert cp.isfinite(output).all()
