import numpy as np

from Tensor import Tensor
from layers.Layer import Layer


class BinaryCrossEntropyLayer(Layer):
    def __init__(self):
        self.ground_truth = None

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements =\
            (- self.ground_truth * np.log(input_tensor.elements)
             - (1 - self.ground_truth) * np.log(1 - input_tensor.elements)).mean()

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = \
            -self.ground_truth / input_tensor.elements + (1 - self.ground_truth) / (1 - input_tensor.elements)
