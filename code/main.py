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
    def forward(self, inputTensor):
        return np.maximum(0, inputTensor)


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


class MnistExample:
    def __init__(self):
        self.network = FullyConnectedNetwork()
        optimizer = StochasticGradientDescent(0.001)

        fcl1 = FullyConnectedLayer(np.random.random(10), np.random.random((10, 784)), optimizer)
        self.network.addLayer(fcl1)

        # sig1 = SigmoidLayer()
        # self.network.addLayer(sig1)
        #
        # fcl2 = FullyConnectedLayer(np.random.random(20), np.random.random((20, 20)), optimizer)
        # self.network.addLayer(fcl2)
        #
        # sig2 = SigmoidLayer()
        # self.network.addLayer(sig2)
        #
        # fcl3 = FullyConnectedLayer(np.random.random(10), np.random.random((10, 20)), optimizer)
        # self.network.addLayer(fcl3)

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

            epoch_error_rates_training.append(np.average(error_rates_training))

            # testing
            error_rates_testing = []

            for i in range(len(test_X)):
                self.errorLayer.setLabel(test_y[i])
                error_rates_testing.append(self.network.forward(test_X[i]))

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

    def test_fcl(self):
        o = StochasticGradientDescent(0.1)
        fc = FullyConnectedLayer(weights=np.array([[0.4047, 0.9563],
                                        [-0.8192, -0.1274],
                                        [0.3662, -0.7252]]).transpose(),
                                 biases=np.array([0, 0]),
                                 optimizer=o)

        # forward test
        mat = fc.forward(np.array([0.4883, 0.5599, 0.5140]))
        np.testing.assert_allclose(mat, np.array([-0.07282827, 0.02287723]))

        # backward test
        mat = fc.backward(np.array([-0.3268, 0.3268]))
        np.testing.assert_allclose(mat, np.array([0.18026288, 0.22608024, -0.35666952]))

        # delta weights test
        np.testing.assert_allclose(fc.deltaBiases, np.array([-0.3268, 0.3268]))

        # delta biases tests
        np.testing.assert_allclose(fc.deltaWeights, np.array([[-0.15957644, 0.15957644],
                                                      [-0.18297532, 0.18297532],
                                                      [-0.1679752,  0.1679752]]))


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
    mnistExample.train(32, train_x_flat, train_y_converted, test_x_flat, test_y_converted)

    print("Running trained network")
    mnistExample.test(test_x_flat[0:128], test_y[0:128])





