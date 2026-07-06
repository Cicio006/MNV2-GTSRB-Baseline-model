
import cupy as cp

from src.layers import  convolution

class CuPyConvolution(convolution.Convolution):
    """CuPy implementation of the two-dimensional convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            dilation, 
            groups, 
            bias
        )

        channels_per_group = in_channels // groups
        scale = cp.sqrt(2.0 / (channels_per_group * kernel_size**2))

        self.weight = (
            cp.random.randn(
                out_channels,
                channels_per_group,
                kernel_size,
                kernel_size,
            ).astype(cp.float32)
            * scale
        )

        self.bias_value = (
            cp.zeros(out_channels, dtype=cp.float32)
            if bias
            else None
        )
        self.inputs: cp.ndarray | None = None
        self.grad_weight = cp.zeros_like(self.weight)
        self.grad_bias = (
            cp.zeros_like(self.bias_value)
            if self.bias_value is not None
            else None
        )

    def forward(self, inputs: cp.ndarray) -> cp.ndarray:
        """Apply a 2D convolution to an NCHW input tensor."""
        if not isinstance(inputs, cp.ndarray):
            raise TypeError("inputs must be a CuPy array")

        if inputs.ndim != 4:
            raise ValueError("inputs must have shape (N, C, H, W)")

        batch_size, channels, height, width = inputs.shape

        if channels != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {channels}"
            )

        effective_kernel = self.dilation * (self.kernel_size - 1) + 1

        output_height = (
            height + 2 * self.padding - effective_kernel
        ) // self.stride + 1

        output_width = (
            width + 2 * self.padding - effective_kernel
        ) // self.stride + 1

        if output_height <= 0 or output_width <= 0:
            raise ValueError("kernel is larger than the padded input")

        self.inputs = inputs
        padded = cp.pad(
            inputs,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )

        output = cp.empty(
            (
                batch_size,
                self.out_channels,
                output_height,
                output_width,
            ),
            dtype=cp.result_type(inputs.dtype, self.weight.dtype),
        )

        input_channels_per_group = self.in_channels // self.groups
        output_channels_per_group = self.out_channels // self.groups

        for group in range(self.groups):
            input_start = group * input_channels_per_group
            input_end = input_start + input_channels_per_group

            output_start = group * output_channels_per_group
            output_end = output_start + output_channels_per_group

            group_input = padded[:, input_start:input_end]

            columns = self._im2col(
                group_input,
                output_height,
                output_width,
            )

            weights = self.weight[output_start:output_end].reshape(
                output_channels_per_group,
                -1,
            )

            group_output = cp.einsum(
                "oc,ncl->nol",
                weights,
                columns,
            )

            output[:, output_start:output_end] = group_output.reshape(
                batch_size,
                output_channels_per_group,
                output_height,
                output_width,
            )

        if self.bias_value is not None:
            output += self.bias_value[None, :, None, None]

        return output
    
    def backward(self, grad: cp.ndarray) -> cp.ndarray:
        """Calculate the gradient of the convolution with respect to its inputs."""
        if not isinstance(grad, cp.ndarray):
            raise TypeError("grad must be a CuPy array")
        if self.inputs is None:
            raise RuntimeError("forward must be called before backward")

        if grad.ndim != 4:
            raise ValueError("grad must have shape (N, C, H, W)")

        batch_size, channels, output_height, output_width = grad.shape

        if channels != self.out_channels:
            raise ValueError(
                f"expected {self.out_channels} channels, got {channels}"
            )

        _, _, input_height, input_width = self.inputs.shape
        padded_inputs = cp.pad(
            self.inputs,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
        )

        padded_input_grad = cp.zeros_like(padded_inputs)
        self.grad_weight = cp.zeros_like(self.weight)

        if self.bias_value is not None:
            self.grad_bias = grad.sum(axis=(0, 2, 3))

        expected_output_height = (
            input_height
            + 2 * self.padding
            - self.dilation * (self.kernel_size - 1)
            - 1
        ) // self.stride + 1
        expected_output_width = (
            input_width
            + 2 * self.padding
            - self.dilation * (self.kernel_size - 1)
            - 1
        ) // self.stride + 1

        if (output_height, output_width) != (
            expected_output_height,
            expected_output_width,
        ):
            raise ValueError(
                "grad spatial shape does not match the previous forward output"
            )

        input_grad_dtype = cp.result_type(grad.dtype, self.weight.dtype)
        padded_input_grad = padded_input_grad.astype(input_grad_dtype, copy=False)

        input_channels_per_group = self.in_channels // self.groups
        output_channels_per_group = self.out_channels // self.groups

        for group in range(self.groups):
            input_start = group * input_channels_per_group
            input_end = input_start + input_channels_per_group

            output_start = group * output_channels_per_group
            output_end = output_start + output_channels_per_group

            group_inputs = padded_inputs[:, input_start:input_end]
            input_columns = self._im2col(
                group_inputs,
                output_height,
                output_width,
            )
            group_grad = grad[:, output_start:output_end].reshape(
                batch_size,
                output_channels_per_group,
                output_height * output_width,
            )
            self.grad_weight[output_start:output_end] = cp.einsum(
                "nol,nkl->ok",
                group_grad,
                input_columns,
            ).reshape(
                output_channels_per_group,
                input_channels_per_group,
                self.kernel_size,
                self.kernel_size,
            )

            weights = self.weight[output_start:output_end].reshape(
                output_channels_per_group,
                -1,
            )
            group_input_columns_grad = cp.einsum(
                "ok,nol->nkl",
                weights,
                group_grad,
            )
            grad_columns = group_input_columns_grad.reshape(
                batch_size,
                input_channels_per_group,
                self.kernel_size,
                self.kernel_size,
                output_height,
                output_width,
            )

            for kernel_y in range(self.kernel_size):
                y_start = kernel_y * self.dilation
                y_end = y_start + output_height * self.stride

                for kernel_x in range(self.kernel_size):
                    x_start = kernel_x * self.dilation
                    x_end = x_start + output_width * self.stride

                    padded_input_grad[
                        :,
                        input_start:input_end,
                        y_start:y_end:self.stride,
                        x_start:x_end:self.stride,
                    ] += grad_columns[:, :, kernel_y, kernel_x]

        if self.padding == 0:
            return padded_input_grad

        return padded_input_grad[
            :,
            :,
            self.padding:-self.padding,
            self.padding:-self.padding,
        ]

    def _im2col(
        self,
        inputs: cp.ndarray,
        output_height: int,
        output_width: int,
    ) -> cp.ndarray:
        batch_size, channels, _, _ = inputs.shape
        kernel = self.kernel_size

        columns = cp.empty(
            (
                batch_size,
                channels,
                kernel,
                kernel,
                output_height,
                output_width,
            ),
            dtype=inputs.dtype,
        )

        for kernel_y in range(kernel):
            for kernel_x in range(kernel):
                y_start = kernel_y * self.dilation
                x_start = kernel_x * self.dilation

                y_end = y_start + output_height * self.stride
                x_end = x_start + output_width * self.stride

                columns[:, :, kernel_y, kernel_x] = inputs[
                    :,
                    :,
                    y_start:y_end:self.stride,
                    x_start:x_end:self.stride,
                ]

        return columns.reshape(
            batch_size,
            channels * kernel * kernel,
            output_height * output_width,
        )
    
    
