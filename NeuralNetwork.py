import numpy as np
from matplotlib import pyplot as plt

from Tensor import Tensor
from Utility import shuffle_in_unison


class NeuronalNetwork:
    def __init__(self, layers, print_every_iteration=False):
        self.layers = layers
        self.tensors = []

        for i in range(len(layers) + 1):  # +1 since input needs a tensor too
            self.tensors.append(Tensor())

        self.print_every_iteration = print_every_iteration

    def forward(self, x):
        self.tensors[0].elements = x

        for i, layer in enumerate(self.layers):
            layer.forward(input_tensor=self.tensors[i], output_tensor=self.tensors[i+1])

    def backprop(self):
        for i, layer in reversed(list(enumerate(self.layers))):
            layer.backward(output_tensor=self.tensors[i+1], input_tensor=self.tensors[i])

    def set_ground_truth(self, ground_truth):
        self.layers[-1].ground_truth = ground_truth

    def get_prediction(self):
        return self.tensors[-2].elements.argmax()

    def get_error(self):
        return self.tensors[-1].elements

    def train(self, epochs, x_train, y_train, x_test, y_test):
        errors_train = []
        errors_test = []
        for i in range(epochs):
            print(f"--- progress: {round(100 * i / epochs,2)}% ---")

            # training one epoch
            epoch_errors_train = self.train_one_epoch(x_train, y_train)
            print(f"training error: {round(epoch_errors_train, 4)}")
            errors_train.append(epoch_errors_train)

            # testing one epoch
            if x_test is not None:
                epoch_errors_test = self.test_one_epoch(x_test, y_test)
                print(f"testing error: {round(epoch_errors_test, 4)}")
                errors_test.append(epoch_errors_test)

                classified_percentage = self.get_correct_classified_percentage(x_test, y_test)
                print(f"testing accuracy: {round(classified_percentage, 2)}%")

        # plot
        plt.plot(range(len(errors_train)), errors_train)
        plt.plot(range(len(errors_test)), errors_test)
        plt.legend({"train", "test"})
        plt.show()

    def train_one_epoch(self, x, y):
        training_errors = []
        correct = 0
        (x, y) = shuffle_in_unison(x, y)
        for j in range(len(x)):
            self.set_ground_truth(y[j])
            self.forward(x[j])
            self.backprop()
            training_errors.append(self.get_error())
            if self.print_every_iteration:
                if self.get_prediction() == y[j].argmax():
                    correct += 1
                print(f"training error: {round(np.average(training_errors), 4)}")
                print(f"training accuracy: {round(100 * correct / (j+1), 2)}%")
        return np.average(training_errors)

    def test_one_epoch(self, x, y):
        epoch_errors = []
        for j in range(len(x)):
            self.set_ground_truth(y[j])
            self.forward(x[j])
            epoch_errors.append(self.get_error())
        return np.average(epoch_errors)

    def get_correct_classified_percentage(self, x, y):
        correct = 0
        for j in range(len(x)):
            self.set_ground_truth(y[j])
            self.forward(x[j])
            if self.get_prediction() == y[j].argmax():
                correct += 1
        return 100 * correct / len(y)
