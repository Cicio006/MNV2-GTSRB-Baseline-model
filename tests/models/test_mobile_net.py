from src.models.Mobile_net import MobileNetV2, ModelShape


class DummyConvolution:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias


class DummyBatchNorm:
    def __init__(self, num_features):
        self.num_features = num_features


class DummyActivation:
    pass


class DummyAveragePooling:
    def __init__(self, kernel_size, stride=1, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class DummyLinear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias


class DummyBackend:
    convolution = DummyConvolution
    batch_norm = DummyBatchNorm
    activation = DummyActivation
    average_pooling = DummyAveragePooling
    linear = DummyLinear


def test_mobile_net_matches_gtsrb_specification():
    model = MobileNetV2(DummyBackend())

    assert len(model.features) == 19
    assert model.output_shape == ModelShape(1280, 4, 4)
    assert model.pooling.kernel_size == 4
    assert model.pooling.stride == 4
    assert model.classifier.in_features == 1280
    assert model.classifier.out_features == 43
