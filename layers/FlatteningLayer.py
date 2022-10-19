from Tensor import Tensor
from layers.Layer import Layer


class FlatteningLayer(Layer):
    def __init__(self):
        self.shape = None

    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = input_tensor.elements.flatten()

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = output_tensor.deltas.reshape(input_tensor.elements.shape)
