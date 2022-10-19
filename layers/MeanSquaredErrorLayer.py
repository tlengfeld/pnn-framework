from Tensor import Tensor
from layers.Layer import Layer


class MeanSquaredErrorLayer(Layer):
    def __init__(self):
        self.ground_truth = None

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = ((input_tensor.elements - self.ground_truth) ** 2).mean()

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = input_tensor.elements - self.ground_truth
