import numpy as np

from Tensor import Tensor
from layers.Layer import Layer


class SigmoidLayer(Layer):
    def __init__(self):
        self.y = 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** -x)

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = self.sigmoid(input_tensor.elements)

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = (output_tensor.elements * (1 - output_tensor.elements)) * output_tensor.deltas
