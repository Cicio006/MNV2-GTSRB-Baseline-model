import pytest


cp = pytest.importorskip("cupy")

from src.backends.cupy.avg_pooling import CuPyAvgPooling


def test_average_pooling_with_overlapping_windows():
    layer = CuPyAvgPooling(kernel_size=2, stride=1)
    inputs = cp.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=cp.float32,
    )

    actual = layer(inputs)
    expected = cp.array(
        [[[[3.0, 4.0], [6.0, 7.0]]]],
        dtype=cp.float32,
    )

    cp.testing.assert_allclose(actual, expected)


def test_average_pooling_excludes_padding_from_mean():
    layer = CuPyAvgPooling(kernel_size=2, stride=1, padding=1)
    inputs = cp.array([[[[2.0]]]], dtype=cp.float32)

    actual = layer(inputs)
    expected = cp.full((1, 1, 2, 2), 2.0, dtype=cp.float32)

    cp.testing.assert_allclose(actual, expected)


def test_average_pooling_backward_matches_overlap_counts():
    layer = CuPyAvgPooling(kernel_size=2, stride=1)
    inputs = cp.ones((1, 1, 3, 3), dtype=cp.float32)
    grad = cp.ones((1, 1, 2, 2), dtype=cp.float32)

    layer(inputs)
    actual = layer.backward(grad)

    expected = cp.array(
        [[[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]],
        dtype=cp.float32,
    )
    cp.testing.assert_allclose(actual, expected)
