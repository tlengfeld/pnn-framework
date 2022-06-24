import unittest

import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from tensorflow.python import tf2


class FullyConnectedNetwork:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, inputTensor):
        for layer in self.layers:
            inputTensor = layer.forward(inputTensor)
        return inputTensor

    def backward(self):
        gradientTensor = None
        for layer in self.layers[::-1]:
            gradientTensor = layer.backward(gradientTensor)
        return gradientTensor


class StochasticGradientDescent:
    def __init__(self, learningRate):
        self.learningRate = learningRate

    def update(self, x, dx):
        return x - self.learningRate * dx


class FullyConnectedLayer:
    def __init__(self, biases, weights, optimizer):
        self.biases = biases
        self.weights = weights
        self.x = None
        self.deltaWeights = None
        self.deltaBiases = None
        self.optimizer = optimizer

    def forward(self, inputTensor):
        self.x = inputTensor
        result = self.weights @ inputTensor + self.biases
        # print(f"forward fcl:\n{result}")
        return result

    def backward(self, gradientTensor):
        result = self.weights.transpose() @ gradientTensor

        self.deltaWeights = self.x[:, np.newaxis] * gradientTensor[np.newaxis, :]
        self.weights = self.optimizer.update(self.weights, self.deltaWeights.transpose())

        self.deltaBiases = gradientTensor
        self.biases = self.optimizer.update(self.biases, self.deltaBiases)

        # print(f"backward fcl:\n{result}")
        # print(f"delta weights:\n{self.deltaWeights}")
        # print(f"delta biases:\n{self.deltaBiases}")
        return result


class SigmoidLayer:
    def __init__(self):
        self.previousActivation = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, previousActivation):
        self.previousActivation = previousActivation
        result = self.sigmoid(previousActivation)
        # print(f"forward sig:\n{result}")
        return result

    def backward(self, gradientTensor):
        r1 = (self.sigmoid(self.previousActivation)).T
        r2 = (1 - self.sigmoid(self.previousActivation))
        result = (r1 * r2) * gradientTensor
        # print(f"backward sig:\n{result}")
        return result


class ReLuLayer:
    def __init__(self):
        self.x = None

    def forward(self, inputTensor):
        self.x = inputTensor
        return np.maximum(0, inputTensor)

    def backward(self, dy):
        return (self.x > 0) * dy


class TanHLayer:
    def forward(self, inputTensor):
        return np.tanh(inputTensor)


class SoftmaxLayer:
    def __init__(self):
        self.y = None

    def softmax(self, x):
        return np.e ** x / (np.e ** x).sum()

    def forward(self, inputTensor):
        self.y = self.softmax(inputTensor)
        # print(f"forward softmax:\n{self.y}")
        return self.y

    def backward(self, gradientTensor):
        r1 = np.matmul(gradientTensor, self.y).sum()
        r2 = (gradientTensor - r1)
        result = np.multiply(self.y, r2)

        # result = np.array([-0.3268, 0.3268])
        # print(f"backward softmax:\n{result}")
        return result


class MeanSquaredErrorLayer:
    def __init__(self):
        self.prediction = None
        self.label = None

    def setLabel(self, label):
        self.label = label

    def forward(self, inputTensor):
        self.prediction = inputTensor
        result = np.sum(0.5 * (inputTensor - self.label) ** 2)
        return result

    def backward(self, _):
        return self.prediction - self.label


class BinaryCrossEntropy:
    def __init__(self):
        self.label = None
        self.prediction = None

    def setLabel(self, label):
        self.label = label

    def forward(self, inputTensor):
        self.prediction = inputTensor
        # errorRate = -np.sum(self.label * np.log(inputTensor))
        errorRate = np.mean(-self.label * np.log(inputTensor) - (1 - self.label) * np.log(1 - inputTensor))
        # print(f"errorRate: {errorRate}")
        return errorRate

    def backward(self, _):
        result = -self.label / self.prediction + (1 - self.label) / (1 - self.prediction)
        # print(f"backward cross-entropy: {result}")
        return result


class CategoricalCrossEntropy:
    def __init__(self):
        self.prediction = None
        self.label = None

    def setLabel(self, label):
        self.label = label

    def forward(self, x):
        self.prediction = x
        return (- self.label * np.log(x)).mean()

    def backward(self, _):
        return - self.label / self.prediction


class ConvolutionLayer:
    def __init__(self, kernels, biases, optimizer):
        self.kernels = kernels
        self.biases = biases
        self.optimizer = optimizer
        self.x = None
        self.weight_updates = None

    def forward(self, input_tensor):
        self.x = input_tensor
        return self.convolution(input_tensor, self.kernels) + self.biases

    def backward(self, dy):
        # calc delta x
        padded_dy = np.pad(dy, pad_width=((self.kernels.shape[0] - 1, self.kernels.shape[0] - 1), (self.kernels.shape[1] - 1, self.kernels.shape[1] - 1), (0, 0)))
        dx = self.convolution(padded_dy, self.calculate_backwards_kernels(self.kernels))

        # weight update
        self.weight_updates = self.channelwise_conv(self.x, dy)
        self.kernels = self.optimizer.update(self.kernels, self.weight_updates)

        bias_updates = np.sum(dy, axis=(0, 1))
        self.biases = self.optimizer.update(self.biases, bias_updates)

        return dx

    def convolution(self, input_tensor, kernels):
        result_x_size = input_tensor.shape[0] - kernels.shape[0] + 1
        result_y_size = input_tensor.shape[1] - kernels.shape[1] + 1
        amount_channels = input_tensor.shape[2]
        amount_filter = kernels.shape[3]

        result = np.zeros((result_x_size, result_y_size, amount_filter))

        for i_channel in range(amount_channels):
            for i_filter in range(amount_filter):
                current_filter = kernels[:, :, i_channel, i_filter]

                for ix in range(result_x_size):
                    for iy in range(result_y_size):
                        input_application_area = input_tensor[ix:ix + kernels.shape[0], iy:iy + kernels.shape[1], i_channel]
                        result[ix, iy, i_filter] += self.inner_product(input_application_area, current_filter)

        return result

    def inner_product(self, a, b):
        assert(a.shape[0] == b.shape[0])
        assert(a.shape[1] == b.shape[1])

        result = 0

        for ix in range(a.shape[0]):
            for iy in range(a.shape[1]):
                result += a[ix, iy] * b[ix, iy]

        return result

    def calculate_backwards_kernels(self, original_kernels):
        return np.rot90(np.transpose(original_kernels, axes=(0, 1, 3, 2)), 2)

    def channelwise_conv(self, x, dy):
        result = np.zeros(self.kernels.size)
        result = result.reshape(self.kernels.shape)

        for i_filter in range(self.kernels.shape[3]):
            current_filter_result = dy[:, :, i_filter]
            current_filter_result = current_filter_result.reshape((dy.shape[0], dy.shape[1], 1, 1))

            for i_channel in range(self.kernels.shape[2]):
                x_current_channel = x[:, :, i_channel]
                x_current_channel = x_current_channel.reshape((x.shape[0], x.shape[1], 1))

                result[:, :, i_channel, i_filter] = self.convolution(x_current_channel, current_filter_result).reshape((self.kernels.shape[0], self.kernels.shape[1]))

        return result


class MaxPoolingLayer:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.input_shape = None
        self.mask = None

    def forward(self, in_tensor):
        self.input_shape = in_tensor.shape

        result_shape = (in_tensor.shape[0] // self.kernel_size[0], in_tensor.shape[1] // self.kernel_size[1])
        result = np.zeros(result_shape)

        self.mask = np.zeros(result.size)
        i_mask = 0

        for ix in range(result.shape[0]):
            for iy in range(result.shape[1]):
                start_x = ix * self.stride[0]
                start_y = iy * self.stride[1]

                application_area = in_tensor[start_x:start_x + self.kernel_size[0], start_y:start_y + self.kernel_size[1]]
                result[ix, iy] = np.max(application_area)

                argmax = self.deflatten_index_2D(np.argmax(application_area), (2, 2))
                flat_argmax = (start_x + argmax[1]) * self.input_shape[0] + (start_y + argmax[0])
                self.mask[i_mask] = flat_argmax
                i_mask += 1

        return result.flatten()

    def backward(self, dy):
        result = np.zeros(self.input_shape)
        i_mask = 0

        for ix in range(dy.shape[0]):
            for iy in range(dy.shape[1]):
                result_x = int(self.mask[i_mask] % self.input_shape[0])
                result_y = int(self.mask[i_mask] // self.input_shape[1])
                result[result_x, result_y] = dy[ix, iy]
                i_mask += 1

        return result

    @staticmethod
    def deflatten_index_2D(flat_index, matrix_size):
        result_x = matrix_size[0] % flat_index
        result_y = matrix_size[1] // flat_index
        return result_x, result_y


class MnistExample:
    def __init__(self):
        self.network = FullyConnectedNetwork()
        optimizer = StochasticGradientDescent(0.1)

        fcl1 = FullyConnectedLayer(np.random.random(20) * 2 - 1, np.random.random((20, 784)) * 2 - 1, optimizer)
        self.network.addLayer(fcl1)

        sig1 = SigmoidLayer()
        self.network.addLayer(sig1)

        fcl2 = FullyConnectedLayer(np.random.random(20) * 2 - 1, np.random.random((20, 20)) * 2 - 1, optimizer)
        self.network.addLayer(fcl2)

        sig2 = SigmoidLayer()
        self.network.addLayer(sig2)

        fcl3 = FullyConnectedLayer(np.random.random(10) * 2 - 1, np.random.random((10, 20)) * 2 - 1, optimizer)
        self.network.addLayer(fcl3)

        softmax = SoftmaxLayer()
        self.network.addLayer(softmax)

        self.errorLayer = MeanSquaredErrorLayer()
        self.network.addLayer(self.errorLayer)

    def train(self, epochSize, train_X, train_y, test_X, test_y):
        # self.find_percentage_of_each_number_in_training_set_converted(train_y)
        epoch_error_rates_training = []
        epoch_error_rates_testing = []

        for j in range(epochSize):
            print(f"Training progress: {100 * j / epochSize}%")

            # training
            (train_X, train_y) = self.shuffle_in_unison(train_X, train_y)

            error_rates_training = []

            for i in range(len(train_X)):
                self.errorLayer.setLabel(train_y[i])
                error_rates_training.append(self.network.forward(train_X[i]))
                self.network.backward()

            print(f"-- Training loss: {np.average(error_rates_training)}")
            epoch_error_rates_training.append(np.average(error_rates_training))

            # testing
            error_rates_testing = []

            for i in range(len(test_X)):
                self.errorLayer.setLabel(test_y[i])
                error_rates_testing.append(self.network.forward(test_X[i]))

            print(f"-- Testing loss: {np.average(error_rates_testing)}")
            epoch_error_rates_testing.append(np.average(error_rates_testing))

        # plotting error rates for training and testing
        plt.plot(range(len(epoch_error_rates_training)), epoch_error_rates_training)
        plt.plot(range(len(epoch_error_rates_testing)), epoch_error_rates_testing)
        plt.legend({"training", "testing"})
        plt.show()

    def shuffle_in_unison(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def find_percentage_of_each_number_in_training_set_converted(self, ground_truths):
        amount_of_each_number = np.zeros(10)

        for ground_truth in ground_truths:
            amount_of_each_number = amount_of_each_number + ground_truth

        print("training set:")
        for i in range(10):
            percentage = amount_of_each_number[i] / np.sum(amount_of_each_number)
            print(f"{i}: {percentage * 100}%")

    def test(self, test_X, test_y):
        right = []
        wrong = []

        for i in range(len(test_X)):
            self.errorLayer.setLabel(test_y[i])
            self.network.forward(test_X[i])
            prediction = self.errorLayer.prediction.argmax()

            if prediction == test_y[i]:
                right.append(prediction)
                # print(f"predicted: {prediction}, actual: {test_y[i]}")
            else:
                wrong.append(prediction)
                # print(f"predicted: {prediction}, actual: {test_y[i]}")

        success_percentage = len(right) / len(test_X)
        print(f"Success percentage: {success_percentage}")


class InternetExample:
    def train(self, iterations):
        network = FullyConnectedNetwork()
        optimizer = StochasticGradientDescent(0.1)

        fcl1 = FullyConnectedLayer(np.array([2, -3]), np.array([[-1, 0.67], [1, -0.67]]), optimizer)
        network.addLayer(fcl1)

        sig1 = SigmoidLayer()
        network.addLayer(sig1)

        fcl2 = FullyConnectedLayer(np.array([1, -4]), np.array([[1, 1], [-0.33, 0.67]]), optimizer)
        network.addLayer(fcl2)

        sig2 = SigmoidLayer()
        network.addLayer(sig2)

        fcl3 = FullyConnectedLayer(np.array([0.5]), np.array([[0.67, -1.3]]), optimizer)
        network.addLayer(fcl3)

        sig3 = SigmoidLayer()
        network.addLayer(sig3)

        errorLayer = BinaryCrossEntropy()
        errorLayer.setLabel(np.array([0]))
        network.addLayer(errorLayer)

        errors = []
        for i in range(iterations):
            errors.append(network.forward(np.array([1, -2])))
            network.backward()

        np.linspace(0, 1, 1000)
        plt.plot(range(100), errors)
        plt.show()


class PnnExample:
    def train(self, iterations):
        network = FullyConnectedNetwork()
        optimizer = StochasticGradientDescent(0.1)

        fcl1 = FullyConnectedLayer(np.array([0.0000, 0.0000, 0.0000]),
                                   np.array([[-0.5057,   0.3987,     -0.8943],
                                             [0.3356,    0.1673,     0.8321],
                                             [-0.3485,   -0.4597,    -0.1121]]).transpose(),
                                   optimizer)
        network.addLayer(fcl1)

        sig1 = SigmoidLayer()
        network.addLayer(sig1)

        fcl2 = FullyConnectedLayer(np.array([0.0000, 0.0000]),
                                   np.array([[0.4047,    0.9563],
                                             [-0.8192,   -0.1274],
                                             [0.3662,    -0.7252]]).transpose(),
                                   optimizer)
        network.addLayer(fcl2)

        softmax = SoftmaxLayer()
        network.addLayer(softmax)

        errorLayer = MeanSquaredErrorLayer()
        errorLayer.setLabel(np.array([0.7095, 0.0942]))
        network.addLayer(errorLayer)

        errors = []
        for i in range(iterations):
            errors.append(network.forward(np.array([0.4183, 0.5209, 0.0291])))
            network.backward()

        np.linspace(0, 1, 1000)
        plt.plot(range(100), errors)
        plt.show()


class QuadraticFunction:
    def plotGradientDescent(self):
        # draw parable
        x = np.linspace(-10, 10, 1000)
        y = x ** 2

        fig, ax = plt.subplots()
        ax.plot(x, y)

        # gradient descent
        x = 6
        gx = None
        learningRate = 0.1

        for i in range(30):
            gx = 2 * x
            x = x - learningRate * gx
            plt.plot(x, x ** 2, marker="x")

        plt.show()


class TestLayers(unittest.TestCase):

    def test_softmax(self):
        s = SoftmaxLayer()

        # forward test
        mat = s.forward(np.array([-0.0728, 0.0229]))
        np.testing.assert_allclose(mat, np.array([0.47609324, 0.52390676]))

        # backward test
        mat = s.backward(np.array([-1.4901, -0.1798]))
        np.testing.assert_allclose(mat, np.array([-0.32682612,  0.32682612]))

    def test_sigmoid(self):
        s = SigmoidLayer()

        # forward test
        mat = s.forward(np.array([-0.0469, 0.2406, 0.0561]))
        np.testing.assert_allclose(mat, np.array([0.48827715, 0.5598615, 0.51402132]))

        # backward test
        mat = s.backward(np.array([0.1803, 0.2261, -0.3567]))
        np.testing.assert_allclose(mat, np.array([0.04505022, 0.05571479, -0.08910487]))

    def test_2D_fcl(self):
        optimizer = StochasticGradientDescent(0.1)
        fcl = FullyConnectedLayer(weights=np.array([[0.4047, 0.9563],
                                        [-0.8192, -0.1274],
                                        [0.3662, -0.7252]]).transpose(),
                                 biases=np.array([0, 0]),
                                 optimizer=optimizer)

        # forward test
        mat = fcl.forward(np.array([0.4883, 0.5599, 0.5140]))
        np.testing.assert_allclose(mat, np.array([-0.07282827, 0.02287723]))

        # backward test
        mat = fcl.backward(np.array([-0.3268, 0.3268]))
        np.testing.assert_allclose(mat, np.array([0.18026288, 0.22608024, -0.35666952]))

        # delta weights test
        np.testing.assert_allclose(fcl.deltaBiases, np.array([-0.3268, 0.3268]))

        # delta biases tests
        np.testing.assert_allclose(fcl.deltaWeights, np.array([[-0.15957644, 0.15957644],
                                                               [-0.18297532, 0.18297532],
                                                               [-0.1679752,  0.1679752]]))

    def test_3D_fcl(self):
        optimizer = StochasticGradientDescent(0.1)
        fcl = FullyConnectedLayer(biases=np.array([0, 0, 0]),
                                 weights=np.array([[-0.5057, 0.3987, -0.8943],
                                                  [0.3356, 0.1673, 0.8321],
                                                  [-0.3485, -0.4597, -0.1121]]).T,
                                 optimizer=optimizer)

        # forward test
        mat = fcl.forward(np.array([0.4183, 0.5209, 0.0291]))
        np.testing.assert_allclose(mat, np.array([-0.0469, 0.2406, 0.0561]), atol=1e-04)

        # backward test
        fcl.backward(np.array([0.0451, 0.0557, -0.0891]))
        np.testing.assert_allclose(fcl.deltaBiases, np.array([0.0451, 0.0557, -0.0891]), atol=1e-04)
        np.testing.assert_allclose(fcl.deltaWeights, np.array([[0.0188, 0.0233, -0.0373],
                                                               [0.0235, 0.0290, -0.0464],
                                                               [0.0013, 0.0016, -0.0026]]), atol=1e-04)

    def test_conv2D_simple(self):
        in_tensor = np.array([0, 6, 1, 1, 6, 7, 9, 3, 7, 7, 3, 5, 3, 8, 3, 8, 6, 8, 0, 2, 4, 3, 2, 9, 1])
        in_tensor = in_tensor.reshape((5, 5, 1))  # (x_size, y_size, amount_channel)

        kernel = np.array([1, 1, 1, 1, -8, 1, 1, 1, 1])
        kernel = kernel.reshape((3, 3, 1, 1))  # (x_size, y_size, amount_channel, amount_filters)

        conv_2d = ConvolutionLayer(kernel)
        actual = conv_2d.forward(in_tensor)

        expected = np.array([-44, 16, -24, 7, 22, -31, -12, -28, 36])
        expected = expected.reshape((3, 3, 1))

        np.testing.assert_allclose(expected, actual, atol=1e-04)

    def test_conv2D_complex(self):
        kernels = np.array([0.1, -0.2, 0.3, 0.4, 0.7, 0.6, 0.9, -1.1, 0.37, -0.9, 0.32, 0.17, 0.9, 0.3, 0.2, -0.7])
        kernels = kernels.reshape((2, 2, 2, 2), order='F')  # (x_size, y_size, amount_channel, amount_filters)

        optimizer = StochasticGradientDescent(0.1)
        conv_2d = ConvolutionLayer(kernels=kernels, biases=np.zeros(2), optimizer=optimizer)

        # forward test
        in_forward = np.array([0.1, -0.2, 0.5, 0.6, 1.2, 1.4, 1.6, 2.2, 0.01, 0.2, -0.3, 4.0, 0.9, 0.3, 0.5, 0.65, 1.1, 0.7, 2.2, 4.4, 3.2, 1.7, 6.3, 8.2])
        in_forward = in_forward.reshape((4, 3, 2), order='F')  # (x_size, y_size, amount_channel)

        expected_forward = np.array([2.0, -0.34000015, -0.8299999, 2.123, -3.8300004, 2.0599995, 1.469, -0.7839999, -1.4639999, -0.12880003, -3.6889997, -1.9839993])
        expected_forward = expected_forward.reshape((3, 2, 2), order='F')

        actual_forward = conv_2d.forward(in_forward)

        np.testing.assert_allclose(expected_forward, actual_forward, atol=1e-04)

        # backward test
        in_backward = np.array([0.1, 0.33, -0.6, -0.25, 1.3, 0.01, -0.5, 0.2, 0.1, -0.8, 0.81, 1.1])
        in_backward = in_backward.reshape((3, 2, 2), order='F')

        expected_backward = np.array([-0.175, 0.537, -0.269, 0.030000009, -0.451, 1.3177, -0.5629999, -1.215, -0.33100003, 0.41320002, 1.0127001, 0.191, -0.38, 0.32099998, -0.072000004, -0.33, -0.905, 1.8259999, 0.997, 0.926, -0.385, 2.1669998, -1.7679999, -0.78099996])
        expected_backward = expected_backward.reshape((4, 3, 2), order='F')

        actual_backward = conv_2d.backward(in_backward)

        np.testing.assert_allclose(expected_backward, actual_backward, atol=1e-04)

        # weight update test
        expected_kernels = np.array([1.18, 1.5369998, -0.12350003, -1.052, 0.54599994, 2.5339997, 0.494, 6.0029993, 1.894, 2.856, -0.33600003, 3.8370001, 1.767, 6.077, 5.557, 13.293001])
        expected_kernels = expected_kernels.reshape((2, 2, 2, 2), order='F')

        np.testing.assert_allclose(expected_kernels, conv_2d.weight_updates, atol=1e-04)

    def test_max_pooling(self):
        max_pooling = MaxPoolingLayer((2, 2))

        in_tensor = np.array([4, 1, 3, 3, 8, 6, 5, 2, 9, 8, 9, 3, 3, 6, 4, 7])
        # in_tensor = np.array([4, 1, 3, 3, 8, 6, 5, 2, 9, 8, 9, 3, 3, 6, 4, 7])
        in_tensor = in_tensor.reshape((4, 4))

        # forward test
        expected_forward = np.array([8, 5, 9, 9])
        actual_forward = max_pooling.forward(in_tensor)

        np.testing.assert_allclose(expected_forward, actual_forward, atol=1e-04)

        # backward test
        expected_backward = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        expected_backward = expected_backward.reshape((4, 4))
        actual_backward = max_pooling.backward(np.array([[1, 1], [1, 1]]))

        np.testing.assert_allclose(expected_backward, actual_backward, atol=1e-04)


if __name__ == "__main__":
    pnnExample = PnnExample()
    # pnnExample.train(100)  # trains only with one input atm

    # -----------------------
    internetExample = InternetExample()
    # internetExample.train(100)  # trains only with one input atm

    # -----------------------
    mnistExample = MnistExample()

    # load mnist data and convert to fitting layout
    print("Loading data")
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_x_flat = train_X.reshape(train_X.shape[0], -1)
    train_x_flat = train_x_flat / 255
    train_y_converted = np.zeros((train_y.size, 10))
    train_y_converted[np.arange(train_y.size), train_y] = 1

    test_x_flat = test_X.reshape(test_X.shape[0], -1)
    test_x_flat = test_x_flat / 255
    test_y_converted = np.zeros((test_y.size, 10))
    test_y_converted[np.arange(test_y.size), test_y] = 1

    print("Training network")
    mnistExample.train(20, train_x_flat[0:1], train_y_converted[0:1], test_x_flat, test_y_converted)

    print("Running trained network")
    mnistExample.test(test_x_flat[0:128], test_y[0:128])





