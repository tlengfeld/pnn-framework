import numpy as np

class FullyConnectedNetwork:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def forward(self, inputTensor):
        for layer in self.layers:
            inputTensor = layer.forward(inputTensor)
        print(f"prediction: {inputTensor}")

    def backward(self):
        gradientTensor = None
        for layer in self.layers[::-1]:
            gradientTensor = layer.backward(gradientTensor)
        print(f"gradientTensor: {gradientTensor}")

class FullyConnectedLayer:
    def __init__(self, biases, weights):
        self.biases = biases
        self.weights = weights

    def forward(self, inputTensor):
        return self.weights @ inputTensor + self.biases

    def backward(self, gradientTensor):
        return self.weights.transpose() @ gradientTensor

class FullyConnectedLayer2:
    def __init__(self, numberInputWeights, numberActivations):
        # self.biases = np.random.rand(numberActivations)
        # self.weights = np.random.rand(numberInputWeights, numberActivations)
        self.biases = np.array([0, 0])
        self.weights = np.array([[1, 1], [0, 0]])

    def forward(self, inputTensor):
        return self.weights @ inputTensor + self.biases

class FullyConnectedLayer3:
    def __init__(self, numberInputWeights, numberActivations):
        # self.biases = np.random.rand(numberActivations)
        # self.weights = np.random.rand(numberInputWeights, numberActivations)
        self.biases = np.array([0, 0, 0])
        self.weights = np.array([[1, 0], [1, 1], [-1, -1]])

    def forward(self, inputTensor):
        return self.weights @ inputTensor + self.biases

class SigmoidLayer:
    def __init__(self):
        self.previousActivation = None

    def sigmoid(self, x):
        return 1 / (1 + np.e**-x)

    def forward(self, previousActivation):
        self.previousActivation = previousActivation
        return self.sigmoid(previousActivation)

    def backward(self, gradientTensor):
        return (self.sigmoid(self.previousActivation) * (1 - self.sigmoid(self.previousActivation))) * gradientTensor

class ReLuLayer:
    def forward(self, inputTensor):
        return np.maximum(0, inputTensor)

class TanHLayer:
    def forward(self, inputTensor):
        return np.tanh(inputTensor)

class SoftmaxLayer:
    def forward(self, inputTensor):
        return np.e**inputTensor / np.sum(np.e**inputTensor)

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
        altErrorRate = -self.expectedTensor * np.log(inputTensor) - (1 - self.expectedTensor) * np.log(1 - inputTensor)
        print(f"errorRate: {altErrorRate}")
        return altErrorRate

    def backward(self, _):
        return -self.expectedTensor / self.prediction + (1 - self.expectedTensor) / (1 - self.prediction)


network = FullyConnectedNetwork()

fcl1 = FullyConnectedLayer(np.array([2, -3]), np.array([[-1, 0.67], [1, -0.67]]))
network.addLayer(fcl1)

sig1 = SigmoidLayer()
network.addLayer(sig1)

fcl2 = FullyConnectedLayer(np.array([1, -4]), np.array([[1, 1], [-0.33, 0.67]]))
network.addLayer(fcl2)

sig2 = SigmoidLayer()
network.addLayer(sig2)

fcl3 = FullyConnectedLayer(np.array([0.5]), np.array([[0.67, -1.3]]))
network.addLayer(fcl3)

sig3 = SigmoidLayer()
network.addLayer(sig3)

expected = np.array([0])
print(f"expected: {expected}")
crossEntropy = CrossEntropy(expected)
network.addLayer(crossEntropy)

network.forward(np.array([1, -2]))
network.backward()


# fcl1 = FullyConnectedLayer2(1, 2)
# fcl2 = FullyConnectedLayer3(2, 3)
# tanh = TanHLayer()
# softm = SoftmaxLayer()
# mean = MeanSquaredErrorLayer(np.array([0.4, 0.4, 0.06]))
# fcn = FullyConnectedNetwork()
# fcn.addLayer(fcl1)
# fcn.addLayer(fcl2)
# fcn.addLayer(tanh)
# fcn.addLayer(softm)
# fcn.addLayer(mean)
# fcn.forward(np.array([1, 1]))
# fcn.backward()

