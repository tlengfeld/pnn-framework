import random

import numpy as np
from keras.datasets import mnist

from NeuralNetwork import NeuronalNetwork
from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.MeanSquaredErrorLayer import MeanSquaredErrorLayer
from layers.SigmoidLayer import SigmoidLayer
from layers.SoftmaxLayer import SoftmaxLayer
from optimizers.GradientDescent import GradientDescent


def train_mnist_fcl_network():
    print("preparing data")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # flatten and normalize
    train_x_flat = train_x.reshape(train_x.shape[0], -1)
    train_x_std = train_x_flat/255

    test_x_flat = test_x.reshape(test_x.shape[0], -1)
    test_x_std = test_x_flat/255

    # result to one hot
    train_y_one_hot = np.zeros((train_y.size, 10))
    train_y_one_hot[np.arange(train_y.size), train_y] = 1

    test_y_one_hot = np.zeros((test_y.size, 10))
    test_y_one_hot[np.arange(test_y.size), test_y] = 1

    # layer-architecture
    hidden_units = 100
    layers = [
        FullyConnectedLayer(weight=np.random.random((hidden_units, train_x_std.shape[1]))*2-1, bias=np.zeros(hidden_units)),
        SigmoidLayer(),
        FullyConnectedLayer(weight=np.random.random((10, hidden_units))*2-1, bias=np.zeros(10)),
        SoftmaxLayer(),
        MeanSquaredErrorLayer()
    ]
    GradientDescent.learning_rate = 0.05  # 0.05 works very well

    network = NeuronalNetwork(layers, print_every_iteration=False)
    print("starting training")
    network.train(epochs=10 , x_train=train_x_std, y_train=train_y_one_hot,
                  x_test=test_x_std, y_test=test_y_one_hot)


if __name__ == '__main__':
    random.seed(1)
    train_mnist_fcl_network()
