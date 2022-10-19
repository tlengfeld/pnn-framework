import math

import numpy as np

from Tensor import Tensor
from layers.Layer import Layer


class PoolingLayer(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.mask = None

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        steps_x = math.floor(input_tensor.get_shape()[0] / self.stride[0])
        steps_y = math.floor(input_tensor.get_shape()[1] / self.stride[1])

        amount_channels = input_tensor.get_shape()[2]

        output_tensor.elements = np.zeros((steps_x, steps_y, amount_channels))
        self.mask = np.zeros((steps_x * steps_y, amount_channels))

        for ix in range(steps_x):  # this only works when stride == kernel_size
            for iy in range(steps_y):  # this only works when stride == kernel_size
                start_x = ix * self.stride[0]
                start_y = iy * self.stride[1]
                matrix = input_tensor.elements[start_x:start_x + self.kernel_size[0], start_y:start_y + self.kernel_size[1]]

                max_value = np.max(matrix, axis=(0, 1))
                output_tensor.elements[ix, iy] = max_value

                for i_channel in range(amount_channels):
                    argmax = np.argmax(matrix[:, :, i_channel])
                    y_max = start_y + argmax % self.kernel_size[1]
                    x_max = start_x + math.floor(argmax / self.kernel_size[1])
                    self.mask[iy + ix * steps_y, i_channel] = y_max + x_max * input_tensor.get_shape()[0]

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = np.zeros(input_tensor.get_shape())

        for channel in range(self.mask.shape[1]):
            for i, mask_index in enumerate(self.mask[:, channel]):
                y = int(mask_index % input_tensor.get_shape()[1])
                x = int(mask_index / input_tensor.get_shape()[1])

                dy_y = i % output_tensor.deltas.shape[1]
                dy_x = math.floor(i / output_tensor.deltas.shape[1])

                input_tensor.deltas[x][y][channel] = output_tensor.deltas[dy_x][dy_y][channel]
