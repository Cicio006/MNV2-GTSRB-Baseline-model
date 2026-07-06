import cupy as cp

from src.layers.avg_pooling import AveragePooling


class CuPyAvgPooling(AveragePooling[cp.ndarray]):
    """CuPy implementation of the average pooling layer."""

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__(kernel_size, stride, padding)
        self.input_shape: tuple[int, int, int, int] | None = None
        self.value_counts: cp.ndarray | None = None

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        """Apply average pooling to an NCHW input tensor."""
        if not isinstance(inputs, cp.ndarray):
            raise TypeError("inputs must be a CuPy array")

        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (N, C, H, W)")

        batch_size, channels, height, width = inputs.shape
        self.input_shape = inputs.shape

        output_height = (
            height + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        output_width = (
            width + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        if output_height <= 0 or output_width <= 0:
            raise ValueError("kernel is larger than the padded input")

        padded_inputs = cp.pad(
            inputs,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )
        valid_values = cp.pad(
            cp.ones((height, width), dtype=cp.float32),
            (
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )

        output_dtype = cp.result_type(inputs.dtype, cp.float32)
        pooled = cp.zeros(
            (batch_size, channels, output_height, output_width),
            dtype=output_dtype,
        )
        value_counts = cp.zeros(
            (output_height, output_width),
            dtype=cp.float32,
        )

        for kernel_y in range(self.kernel_size):
            y_end = kernel_y + output_height * self.stride

            for kernel_x in range(self.kernel_size):
                x_end = kernel_x + output_width * self.stride

                pooled += padded_inputs[
                    :,
                    :,
                    kernel_y:y_end:self.stride,
                    kernel_x:x_end:self.stride,
                ]
                value_counts += valid_values[
                    kernel_y:y_end:self.stride,
                    kernel_x:x_end:self.stride,
                ]

        self.value_counts = value_counts
        return pooled / value_counts[None, None, :, :]
    
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Calculate the gradient of the average pooling layer with respect to its inputs."""
        if not isinstance(grad, cp.ndarray):
            raise TypeError("grad must be a CuPy array")
        if self.input_shape is None or self.value_counts is None:
            raise RuntimeError("forward must be called before backward")

        if grad.ndim != 4:
            raise ValueError("grad must have shape (N, C, H, W)")

        batch_size, channels, output_height, output_width = grad.shape
        _, _, input_height, input_width = self.input_shape

        padded_grad_input = cp.zeros(
            (
                batch_size,
                channels,
                input_height + 2 * self.padding,
                input_width + 2 * self.padding,
            ),
            dtype=cp.result_type(grad.dtype, cp.float32),
        )
        scaled_grad = grad / self.value_counts[None, None, :, :]

        for kernel_y in range(self.kernel_size):
            y_end = kernel_y + output_height * self.stride

            for kernel_x in range(self.kernel_size):
                x_end = kernel_x + output_width * self.stride

                padded_grad_input[
                    :,
                    :,
                    kernel_y:y_end:self.stride,
                    kernel_x:x_end:self.stride,
                ] += scaled_grad

        if self.padding == 0:
            return padded_grad_input

        return padded_grad_input[
            :,
            :,
            self.padding:-self.padding,
            self.padding:-self.padding,
        ]
