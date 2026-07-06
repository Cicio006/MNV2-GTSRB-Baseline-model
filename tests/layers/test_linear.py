import pytest

from src.layers.linear import Linear


class DummyLinear(Linear[object]):
    def forward(self, inputs: object) -> object:
        return inputs

    def backward(self, grad: object) -> object:
        return grad


def test_linear_is_abstract():
    with pytest.raises(TypeError):
        Linear(in_features=4, out_features=2)


def test_linear_stores_configuration():
    layer = DummyLinear(in_features=4, out_features=2, bias=False)

    assert layer.in_features == 4
    assert layer.out_features == 2
    assert layer.bias is False


def test_linear_call_delegates_to_forward():
    layer = DummyLinear(in_features=4, out_features=2)
    inputs = object()

    assert layer(inputs) is inputs


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("in_features", 0),
        ("out_features", 0),
    ],
)
def test_linear_rejects_invalid_feature_counts(argument, value):
    arguments = {
        "in_features": 4,
        "out_features": 2,
    }
    arguments[argument] = value

    with pytest.raises(ValueError):
        DummyLinear(**arguments)


def test_linear_bias_must_be_boolean():
    with pytest.raises(TypeError):
        DummyLinear(in_features=4, out_features=2, bias=1)
