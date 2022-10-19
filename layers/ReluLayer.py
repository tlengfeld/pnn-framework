import numpy as np

from Tensor import Tensor
from layers.Layer import Layer


class ReluLayer(Layer):
    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = np.maximum(0, input_tensor.elements)

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = (input_tensor.elements > 0) * output_tensor.deltas
