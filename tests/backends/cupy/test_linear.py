import pytest


cp = pytest.importorskip("cupy")

from src.backends.cupy.linear import CuPyLinear


def test_linear_values_with_bias():
    layer = CuPyLinear(in_features=3, out_features=2, bias=True)
    layer.weight = cp.array(
        [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]],
        dtype=cp.float32,
    )
    layer.bias_value = cp.array([0.5, -0.5], dtype=cp.float32)
    inputs = cp.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=cp.float32,
    )

    actual = layer(inputs)
    expected = cp.array(
        [[14.5, 1.5], [32.5, 1.5]],
        dtype=cp.float32,
    )

    cp.testing.assert_allclose(actual, expected)


def test_linear_values_without_bias():
    layer = CuPyLinear(in_features=2, out_features=1, bias=False)
    layer.weight = cp.array([[2.0, -1.0]], dtype=cp.float32)
    inputs = cp.array([[3.0, 4.0]], dtype=cp.float32)

    actual = layer(inputs)
    expected = cp.array([[2.0]], dtype=cp.float32)

    cp.testing.assert_allclose(actual, expected)
    assert layer.bias_value is None


def test_linear_supports_additional_batch_dimensions():
    layer = CuPyLinear(in_features=3, out_features=2)
    inputs = cp.zeros((2, 4, 3), dtype=cp.float32)

    output = layer(inputs)

    assert output.shape == (2, 4, 2)


def test_linear_rejects_wrong_feature_count():
    layer = CuPyLinear(in_features=3, out_features=2)
    inputs = cp.zeros((2, 4), dtype=cp.float32)

    with pytest.raises(ValueError, match="expected 3 input features"):
        layer(inputs)


def test_linear_backward_values_with_bias():
    layer = CuPyLinear(in_features=2, out_features=3, bias=True)
    layer.weight = cp.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=cp.float32,
    )
    inputs = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    grad = cp.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]], dtype=cp.float32)

    layer(inputs)
    grad_input = layer.backward(grad)

    cp.testing.assert_allclose(
        grad_input,
        cp.array([[-4.0, -4.0], [5.0, 8.0]], dtype=cp.float32),
    )
    cp.testing.assert_allclose(
        layer.grad_weight,
        cp.array([[7.0, 10.0], [3.0, 4.0], [-1.0, -2.0]], dtype=cp.float32),
    )
    cp.testing.assert_allclose(
        layer.grad_bias,
        cp.array([3.0, 1.0, -1.0], dtype=cp.float32),
    )
