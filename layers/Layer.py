from Tensor import Tensor


class Layer:
    def forward(self, input_tensor: Tensor, output_tensor: Tensor):
        output_tensor.elements = input_tensor.elements

    def backward(self, output_tensor: Tensor, input_tensor: Tensor):
        input_tensor.deltas = output_tensor.deltas

