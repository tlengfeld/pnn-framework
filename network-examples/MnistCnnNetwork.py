import random

import numpy as np
from keras.datasets import mnist

from NeuralNetwork import NeuronalNetwork
from layers.CategoricalCrossEntropyLayer import CategoricalCrossEntropyLayer
from layers.Conv2DLayer import Conv2DLayer
from layers.FlatteningLayer import FlatteningLayer
from layers.FullyConnectedLayer import FullyConnectedLayer
from layers.PoolingLayer import PoolingLayer
from layers.SigmoidLayer import SigmoidLayer
from layers.SoftmaxLayer import SoftmaxLayer
from optimizers.GradientDescent import GradientDescent


def train_mnist_cnn_network():
    print("preparing data")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # add channel dimension and normalize to train
    train_x_ch = np.expand_dims(train_x, axis=3)
    train_x_norm = train_x_ch / 255

    test_x_ch = np.expand_dims(test_x, axis=3)
    test_x_norm = test_x_ch / 255

    # result to one hot
    train_y_one_hot = np.zeros((train_y.size, 10))
    train_y_one_hot[np.arange(train_y.size), train_y] = 1

    test_y_one_hot = np.zeros((test_y.size, 10))
    test_y_one_hot[np.arange(test_y.size), test_y] = 1

    layers = [
        Conv2DLayer(kernels=np.random.random((20, 20, 1, 5))*2-1, bias=np.zeros(5)),
        PoolingLayer(kernel_size=(4, 4, 5)),
        FlatteningLayer(),
        FullyConnectedLayer(weight=np.random.random((40, 20))*2-1, bias=np.zeros(40)),
        SigmoidLayer(),
        FullyConnectedLayer(weight=np.random.random((10, 40)), bias=np.zeros(10)),
        SoftmaxLayer(),
        CategoricalCrossEntropyLayer()
    ]
    GradientDescent.learning_rate = 0.05

    network = NeuronalNetwork(layers, print_every_iteration=True)
    random.seed(1)
    print("starting training")
    network.train(epochs=5, x_train=train_x_norm[:10000], y_train=train_y_one_hot[:10000],
                  x_test=test_x_norm[:100], y_test=test_y_one_hot[:100])


if __name__ == '__main__':
    train_mnist_cnn_network()
