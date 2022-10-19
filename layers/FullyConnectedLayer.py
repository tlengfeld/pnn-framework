import numpy as np

from Tensor import Tensor
from layers.Layer import Layer
from optimizers.GradientDescent import GradientDescent


class FullyConnectedLayer(Layer):
    def __init__(self, weight, bias):
        self.delta_weights = None
        self.delta_biases = None
        self.weight = weight
        self.bias = bias

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = self.weight @ input_tensor.elements + self.bias

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = self.weight.T @ output_tensor.deltas

        self.delta_biases = output_tensor.deltas
        self.bias = GradientDescent.step(self.bias, self.delta_biases)

        self.delta_weights = np.expand_dims(output_tensor.deltas, 1) * np.expand_dims(input_tensor.elements, 0)

        self.weight = GradientDescent.step(self.weight, self.delta_weights)
