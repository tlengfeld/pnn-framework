import numpy as np

from Tensor import Tensor
from layers.Layer import Layer


class SoftmaxLayer(Layer):
    @staticmethod
    def softmax(x):
        return np.e ** x / (np.e ** x).sum()

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = self.softmax(input_tensor.elements)

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = \
            output_tensor.elements * \
            (output_tensor.deltas - (output_tensor.deltas * output_tensor.elements).sum())