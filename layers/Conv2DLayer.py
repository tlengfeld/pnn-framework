import numpy as np

from Utility import inner_product
from layers.Layer import Layer
from optimizers.GradientDescent import GradientDescent


class Conv2DLayer(Layer):
    def __init__(self, kernels, bias):
        self.kernels = kernels
        self.bias = bias

        # for testing reasons
        self.delta_weights = None
        self.modifiedKernel = None

    @staticmethod
    def add_padding(matrix, padding):
        return np.pad(matrix, pad_width=((padding[0], padding[0]), (padding[1], padding[1]), (0, 0)))

    @staticmethod
    def convolution(image, kernel):
        result_x_size = image.shape[0] - kernel.shape[0] + 1
        result_y_size = image.shape[1] - kernel.shape[1] + 1
        amount_filter = kernel.shape[3]

        result = np.empty((result_x_size, result_y_size, amount_filter))

        for i_filter in range(amount_filter):
            current_filter = kernel[:, :, :, i_filter]

            for ix in range(result_x_size):
                for iy in range(result_y_size):
                    input_application_area = image[ix:ix + kernel.shape[0], iy:iy + kernel.shape[1]]
                    result[ix, iy, i_filter] = inner_product(input_application_area, current_filter)

        return result

    def channel_wise_convolution(self, x, dy):
        result = np.zeros(self.kernels.shape)

        for kernel in range(dy.shape[2]):
            for channel in range(x.shape[2]):
                x_ci = np.expand_dims(x[:, :, channel], axis=2)
                dy_ci = np.expand_dims(dy[:, :, kernel], axis=(2, 3))
                conv_ci = self.convolution(x_ci, dy_ci)[:, :, 0]
                result[:, :, channel, kernel] = conv_ci

        return result

    def forward(self, input_tensor, output_tensor):
        output_tensor.elements = self.convolution(input_tensor.elements, self.kernels) + self.bias

    def backward(self, output_tensor, input_tensor):
        self.modifiedKernel = np.rot90(np.transpose(self.kernels, axes=(0, 1, 3, 2)), k=2)
        dy_with_padding = self.add_padding(output_tensor.deltas, [self.kernels.shape[0]-1, self.kernels.shape[1]-1])

        # weight update
        self.delta_weights = self.channel_wise_convolution(input_tensor.elements, output_tensor.deltas)
        self.kernels = GradientDescent.step(self.kernels, self.delta_weights)

        # bias update
        delta_biases = np.sum(output_tensor.deltas, axis=(0, 1))
        self.bias = GradientDescent.step(self.bias, delta_biases)  # not tested

        input_tensor.deltas = self.convolution(dy_with_padding, self.modifiedKernel)

