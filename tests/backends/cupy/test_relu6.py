import pytest


cp = pytest.importorskip("cupy")

from src.backends.cupy.activation import CuPyReLU6


def test_relu6_values():
    inputs = cp.array([-2.0, 0.0, 2.5, 6.0, 8.0])
    expected = cp.array([0.0, 0.0, 2.5, 6.0, 6.0])

    actual = CuPyReLU6()(inputs)

    cp.testing.assert_array_equal(actual, expected)


def test_relu6_backward_uses_forward_input_mask():
    layer = CuPyReLU6()
    inputs = cp.array([-2.0, 0.0, 2.5, 6.0, 8.0], dtype=cp.float32)
    grad = cp.ones_like(inputs)

    layer(inputs)
    actual = layer.backward(grad)

    expected = cp.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=cp.float32)
    cp.testing.assert_array_equal(actual, expected)
