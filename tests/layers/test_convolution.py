import pytest


from src.layers.convolution import Convolution


class DummyConvolution(Convolution[object]):
    """Minimal concrete implementation used to test the abstract interface."""

    def forward(self, inputs: object) -> object:
        return inputs

    def backward(self, grad: object) -> object:
        return grad


class SpyConvolution(Convolution[object]):
    """Records how the abstract layer delegates to the backend implementation."""

    def __init__(self) -> None:
        super().__init__(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
        )
        self.forward_called = False
        self.received_inputs = None

    def forward(self, inputs: object) -> str:
        self.forward_called = True
        self.received_inputs = inputs
        return "backend result"

    def backward(self, grad: object) -> object:
        return grad


def test_convolution_is_abstract():
    with pytest.raises(TypeError):
        Convolution(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
        )


def test_convolution_stores_configuration():
    layer = DummyConvolution(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=2,
        groups=2,
        bias=True,
    )

    assert layer.in_channels == 4
    assert layer.out_channels == 8
    assert layer.kernel_size == 3
    assert layer.stride == 2
    assert layer.padding == 1
    assert layer.dilation == 2
    assert layer.groups == 2
    assert layer.bias is True


def test_call_delegates_to_forward():
    layer = SpyConvolution()
    inputs = object()

    result = layer(inputs)

    assert layer.forward_called is True
    assert layer.received_inputs is inputs
    assert result == "backend result"

@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("in_channels", 0),
        ("out_channels", 0),
        ("kernel_size", 0),
        ("stride", 0),
        ("padding", -1),
        ("dilation", 0),
        ("groups", 0),
    ],
)
def test_invalid_numeric_arguments_raise_value_error(argument, value):
    arguments = {
        "in_channels": 4,
        "out_channels": 8,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": False,
    }
    arguments[argument] = value

    with pytest.raises(ValueError):
        DummyConvolution(**arguments)


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "groups"),
    [
        (3, 8, 2),
        (4, 9, 2),
    ],
)
def test_channels_must_be_divisible_by_groups(
    in_channels,
    out_channels,
    groups,
):
    with pytest.raises(ValueError):
        DummyConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            groups=groups,
        )


def test_bias_must_be_boolean():
    with pytest.raises(TypeError):
        DummyConvolution(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            bias=1,
        )
