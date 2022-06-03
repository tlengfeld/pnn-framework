import unittest

import numpy as np

class FullyConnectedNetwork:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, inputTensor):
        for layer in self.layers:
            inputTensor = layer.forward(inputTensor)

    def backward(self):
        gradientTensor = None
        for layer in self.layers[::-1]:
            gradientTensor = layer.backward(gradientTensor)

class FullyConnectedLayer:
    def __init__(self, biases, weights):
        self.biases = biases
        self.weights = weights
        self.x = None
        self.deltaWeights = None
        self.deltaBiases = None

    def forward(self, inputTensor):
        self.x = inputTensor
        result = self.weights @ inputTensor + self.biases
        print(f"forward fcl:\n{result}")
        return result

    def backward(self, gradientTensor):
        result = self.weights.transpose() @ gradientTensor
        self.deltaWeights = self.x[:, np.newaxis] * gradientTensor[np.newaxis, :]
        self.deltaBiases = gradientTensor
        print(f"backward fcl:\n{result}")
        print(f"delta weights:\n{self.deltaWeights}")
        print(f"delta biases:\n{self.deltaBiases}")
        return result

class SigmoidLayer:
    def __init__(self):
        self.previousActivation = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, previousActivation):
        self.previousActivation = previousActivation
        result = self.sigmoid(previousActivation)
        print(f"forward sig:\n{result}")
        return result

    def backward(self, gradientTensor):
        r1 = (self.sigmoid(self.previousActivation)).T
        r2 = (1 - self.sigmoid(self.previousActivation))
        result = (r1 * r2) * gradientTensor
        print(f"backward sig:\n{result}")
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
        print(f"forward softmax:\n{self.y}")
        return self.y

    def backward(self, gradientTensor):
        r1 = np.matmul(gradientTensor, self.y).sum()
        r2 = (gradientTensor - r1)
        result = np.multiply(self.y, r2)

        # result = np.array([-0.3268, 0.3268])
        print(f"backward softmax:\n{result}")
        return result

class MeanSquaredErrorLayer:
    def __init__(self, resultTensor):
        self.inputTensor = None
        self.resultTensor = resultTensor

    def forward(self, inputTensor):
        self.inputTensor = inputTensor
        return np.sum(0.5 * (inputTensor - self.resultTensor)**2)

    def backward(self, _):
        return self.inputTensor - self.resultTensor


class CrossEntropy:
    def __init__(self, expectedTensor):
        self.expectedTensor = expectedTensor
        self.prediction = None

    def forward(self, inputTensor):
        self.prediction = inputTensor
        # errorRate = -np.sum(self.expectedTensor * np.log(inputTensor))
        errorRate = -self.expectedTensor * np.log(inputTensor) - (1 - self.expectedTensor) * np.log(1 - inputTensor)
        print(f"errorRate: {errorRate}")
        return errorRate

    def backward(self, _):
        result = -self.expectedTensor / self.prediction + (1 - self.expectedTensor) / (1 - self.prediction)
        print(f"backward cross-entropy: {result}")
        return result


# example from the internet
# network = FullyConnectedNetwork()
#
# fcl1 = FullyConnectedLayer(np.array([2, -3]), np.array([[-1, 0.67], [1, -0.67]]))
# network.addLayer(fcl1)
#
# sig1 = SigmoidLayer()
# network.addLayer(sig1)
#
# fcl2 = FullyConnectedLayer(np.array([1, -4]), np.array([[1, 1], [-0.33, 0.67]]))
# network.addLayer(fcl2)
#
# sig2 = SigmoidLayer()
# network.addLayer(sig2)
#
# fcl3 = FullyConnectedLayer(np.array([0.5]), np.array([[0.67, -1.3]]))
# network.addLayer(fcl3)
#
# sig3 = SigmoidLayer()
# network.addLayer(sig3)
#
# expected = np.array([0])
# print(f"expected: {expected}")
# crossEntropy = CrossEntropy(expected)
# network.addLayer(crossEntropy)
#
# network.forward(np.array([1, -2]))
# network.backward()

# example from pnn
network = FullyConnectedNetwork()

fcl1 = FullyConnectedLayer(np.array([0.0000, 0.0000, 0.0000]),
                           np.array([[-0.5057,   0.3987,     -0.8943],
                                     [0.3356,    0.1673,     0.8321],
                                     [-0.3485,   -0.4597,    -0.1121]]).transpose())
network.addLayer(fcl1)

sig1 = SigmoidLayer()
network.addLayer(sig1)

fcl2 = FullyConnectedLayer(np.array([0.0000, 0.0000]),
                           np.array([[0.4047,    0.9563],
                                     [-0.8192,   -0.1274],
                                     [0.3662,    -0.7252]]).transpose())
network.addLayer(fcl2)

softmax = SoftmaxLayer()
network.addLayer(softmax)

expected = np.array([0.7095, 0.0942])
crossEntropy = MeanSquaredErrorLayer(expected)
network.addLayer(crossEntropy)

network.forward(np.array([0.4183, 0.5209, 0.0291]))
print(f"expected: {expected}")

network.backward()


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
        fc = FullyConnectedLayer(weights=np.array([[0.4047, 0.9563],
                                        [-0.8192, -0.1274],
                                        [0.3662, -0.7252]]).transpose(),
                                 biases=np.array([0, 0]))

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
    unittest.main()
