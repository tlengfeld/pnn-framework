import unittest

import numpy as np
import numpy.testing

from Tensor import Tensor
from layers.Conv2DLayer import Conv2DLayer
from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.PoolingLayer import PoolingLayer
from layers.SigmoidLayer import SigmoidLayer
from layers.SoftmaxLayer import SoftmaxLayer

if __name__ == '__main__':
    unittest.main()


class TestNNLayers(unittest.TestCase):

    def test_softmax(self):
        softmax = SoftmaxLayer()

        input_tensor = Tensor(elements=np.array([-0.0728, 0.0229]))
        output_tensor = Tensor(deltas=np.array([-1.4901, -0.1798]))

        # forward test
        softmax.forward(input_tensor, output_tensor)
        numpy.testing.assert_allclose(np.array([0.47609324, 0.52390676]), output_tensor.elements)

        # backward test
        softmax.backward(output_tensor, input_tensor)
        numpy.testing.assert_allclose(np.array([-0.32682612, 0.32682612]), input_tensor.deltas)

    def test_sigmoid(self):
        sigmoid = SigmoidLayer()

        input_tensor = Tensor(elements=np.array([-0.0469, 0.2406, 0.0561]))
        output_tensor = Tensor(deltas=np.array([0.1803, 0.2261, -0.3567]))

        # forward test
        sigmoid.forward(input_tensor, output_tensor)
        numpy.testing.assert_allclose(np.array([0.48827715, 0.5598615, 0.51402132]), output_tensor.elements)

        # backward test
        sigmoid.backward(output_tensor, input_tensor)
        numpy.testing.assert_allclose(np.array([0.04505022, 0.05571479, -0.08910487]), input_tensor.deltas)

    def test_fully_connected(self):
        fully_connected = FullyConnectedLayer(weight=np.array([[0.4047, 0.9563],
                                                               [-0.8192, -0.1274],
                                                               [0.3662, -0.7252]]).T,
                                              bias=np.array([0, 0]))

        input_tensor = Tensor(elements=np.array([0.4883, 0.5599, 0.5140]))
        output_tensor = Tensor(deltas=np.array([-0.3268, 0.3268]))

        # forward test
        fully_connected.forward(input_tensor, output_tensor)
        numpy.testing.assert_allclose(np.array([-0.07282827, 0.02287723]), output_tensor.elements)

        # backward test
        fully_connected.backward(output_tensor, input_tensor)
        numpy.testing.assert_allclose(np.array([0.18026288, 0.22608024, -0.35666952]), input_tensor.deltas)
        numpy.testing.assert_allclose(np.array([-0.3268, 0.3268]), fully_connected.delta_biases)
        numpy.testing.assert_allclose(np.array([[-0.15957644, 0.15957644],
                                                [-0.18297532, 0.18297532],
                                                [-0.1679752, 0.1679752]]).T, fully_connected.delta_weights)

    def test_fully_connected2(self):
        fully_connected = FullyConnectedLayer(weight=np.array([[-0.5057, 0.3987, -0.8943],
                                                               [0.3356, 0.1673, 0.8321],
                                                               [-0.3485, -0.4597, -0.1121]]).T,
                                              bias=np.array([0, 0, 0]))

        input_tensor = Tensor(elements=np.array([0.4183, 0.5209, 0.0291]))
        output_tensor = Tensor(deltas=np.array([0.0451, 0.0557, -0.0891]))

        # forward test
        fully_connected.forward(input_tensor, output_tensor)
        numpy.testing.assert_allclose(np.array([-0.0469, 0.2406, 0.0561]), output_tensor.elements, atol=1e-04)

        # backward test
        fully_connected.backward(output_tensor, input_tensor)
        numpy.testing.assert_allclose(np.array([0.0451, 0.0557, -0.0891]), fully_connected.delta_biases, atol=1e-04)
        numpy.testing.assert_allclose(np.array([[0.0188, 0.0233, -0.0373],
                                                [0.0235, 0.0290, -0.0464],
                                                [0.0013, 0.0016, -0.0026]]).T,
                                      fully_connected.delta_weights, atol=1e-04)

    # without channels
    def test_conv2D_simple(self):
        input_elements = np.array([0, 6, 1, 1, 6, 7, 9, 3, 7, 7, 3, 5, 3, 8, 3, 8, 6, 8, 0, 2, 4, 3, 2, 9, 1])\
            .reshape((5, 5, 1))  # (x_size, y_size, amount_channel)
        input_tensor = Tensor(elements=input_elements)

        output_tensor = Tensor()

        expected_elements = np.array([-44, 16, -24, 7, 22, -31, -12, -28, 36]).reshape((3, 3, 1))
        expected_tensor = Tensor(elements=expected_elements)

        kernel = np.array([1, 1, 1, 1, -8, 1, 1, 1, 1])
        kernel = kernel.reshape((3, 3, 1, 1))  # (x_size, y_size, amount_channel, amount_filters)

        conv_2d = Conv2DLayer(kernels=kernel, bias=np.array([0]))
        conv_2d.forward(input_tensor, output_tensor)

        np.testing.assert_allclose(expected_tensor.elements, output_tensor.elements, atol=1e-04)

    def test_conv2DLayer(self):
        kernel = np.array([0.1, -0.2, 0.3, 0.4, 0.7, 0.6, 0.9, -1.1, 0.37, -0.9, 0.32, 0.17, 0.9, 0.3, 0.2, -0.7])\
            .reshape((2, 2, 2, 2), order='F')

        input_elements = np.array([0.1, -0.2, 0.5, 0.6, 1.2, 1.4, 1.6, 2.2, 0.01, 0.2, -0.3, 4.0, 0.9, 0.3, 0.5, 0.65, 1.1, 0.7, 2.2, 4.4, 3.2, 1.7, 6.3, 8.2])\
            .reshape((4, 3, 2), order='F')
        input_tensor = Tensor(elements=input_elements)

        output_deltas = np.array([0.1, 0.33, -0.6, -0.25, 1.3, 0.01, -0.5, 0.2, 0.1, -0.8, 0.81, 1.1])\
            .reshape((3, 2, 2), order='F')
        output_tensor = Tensor(deltas=output_deltas)

        conv = Conv2DLayer(kernels=kernel, bias=np.array([0, 0]))

        # forward
        expected_y = np.array([2.0, -0.34000015, -0.8299999, 2.123, -3.8300004, 2.0599995, 1.469, -0.7839999, -1.4639999, -0.12880003, -3.6889997, -1.9839993])\
            .reshape((3, 2, 2), order='F')

        conv.forward(input_tensor, output_tensor)

        numpy.testing.assert_allclose(expected_y, output_tensor.elements, atol=1e-06)

        # backward
        expected_modified_kernel = np.array([0.4, 0.3, -0.2, 0.1, 0.17, 0.32, -0.9, 0.37, -1.1, 0.9, 0.6, 0.7, -0.7, 0.2, 0.3, 0.9])\
            .reshape((2, 2, 2, 2), order='F')
        expected_dx = np.array([-0.175, 0.537, -0.269, 0.030000009, -0.451, 1.3177, -0.5629999, -1.215, -0.33100003, 0.41320002, 1.0127001, 0.191, -0.38, 0.32099998, -0.072000004, -0.33, -0.905, 1.8259999, 0.997, 0.926, -0.385, 2.1669998, -1.7679999, -0.78099996])\
            .reshape((4, 3, 2), order='F')
        expected_dw = np.array([1.18, 1.5369998, -0.12350003, -1.052, 0.54599994, 2.5339997, 0.494, 6.0029993, 1.894, 2.856, -0.33600003, 3.8370001, 1.767, 6.077, 5.557, 13.293001])\
            .reshape((2, 2, 2, 2), order='F')

        conv.backward(output_tensor, input_tensor)

        numpy.testing.assert_allclose(expected_modified_kernel, conv.modifiedKernel, atol=1e-04)
        numpy.testing.assert_allclose(expected_dx, input_tensor.deltas, atol=1e-04)
        numpy.testing.assert_allclose(expected_dw, conv.delta_weights, atol=1e-04)

    def test_pooling(self):
        image = np.array([[4, 1, 1, 3, 3, 4],
                          [8, 6, 0, 5, 2, 2],
                          [0, 6, 9, 5, 1, 8],
                          [4, 8, 5, 7, 4, 2],
                          [9, 8, 5, 9, 3, 0],
                          [3, 6, 7, 4, 7, 5]])
        result = np.array([[8, 5, 4],
                           [8, 9, 8],
                           [9, 9, 7]])
        dy = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        backward_result = np.array([[0, 0, 0, 0, 0, 3],
                                    [1, 0, 0, 2, 0, 0],
                                    [0, 0, 5, 0, 0, 6],
                                    [0, 4, 0, 0, 0, 0],
                                    [7, 0, 0, 8, 0, 0],
                                    [0, 0, 0, 0, 9, 0]])

        pooling = PoolingLayer(kernel_size=(2, 2))
        input_tensor = Tensor(elements=np.expand_dims(image, axis=2))
        output_tensor = Tensor(deltas=np.expand_dims(dy, axis=2))

        # forward
        pooling.forward(input_tensor, output_tensor)
        numpy.testing.assert_equal(np.expand_dims(result, axis=2), output_tensor.elements)
        numpy.testing.assert_equal(np.array([[6, 9, 5, 19, 14, 17, 24, 27, 34]]).T, pooling.mask)

        # backward
        pooling.backward(output_tensor, input_tensor)
        backward_result = np.expand_dims(backward_result, axis=2)
        numpy.testing.assert_equal(backward_result, input_tensor.deltas)

    def test_pooling_multiple_channel(self):
        input_elements = np.array([[[4, 4], [1, 8], [3, 9], [3, 3]],
                                   [[8, 1], [6, 6], [5, 8], [2, 6]],
                                   [[9, 3], [8, 5], [9, 9], [3, 4]],
                                   [[3, 3], [6, 2], [4, 3], [7, 7]]])

        expected_forward = np.array([[[8, 8], [5, 9]],
                                     [[9, 5], [9, 9]]])

        dy = np.array([[[1, 5], [2, 6]],
                       [[3, 7], [4, 8]]])

        expected_backward = np.array([[[0, 0], [0, 5], [0, 6], [0, 0]],
                                      [[1, 0], [0, 0], [2, 0], [0, 0]],
                                      [[3, 0], [0, 7], [4, 8], [0, 0]],
                                      [[0, 0], [0, 0], [0, 0], [0, 0]]])

        input_tensor = Tensor(elements=input_elements)
        output_tensor = Tensor(deltas=dy)
        pooling = PoolingLayer(kernel_size=(2, 2))

        # forward test
        pooling.forward(input_tensor, output_tensor)
        np.testing.assert_equal(expected_forward, output_tensor.elements)

        # backward test
        pooling.backward(output_tensor, input_tensor)
        np.testing.assert_equal(expected_backward, input_tensor.deltas)
