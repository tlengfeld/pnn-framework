import numpy as np

# this method prints a plot with a parable and applies gd on it
from matplotlib import pyplot as plt

from optimizers.GradientDescent import GradientDescent


def plot_on_parable():
    GradientDescent.learning_rate = 0.1

    # parable
    x = np.arange(-5, 5, 0.1)
    y = x ** 2
    plt.plot(x, y)

    xi = 5
    yi = 5 ** 2

    for i in range(100):
        plt.plot(xi, yi, marker="o")
        dx = 2 * xi
        xi = GradientDescent.step(xi, dx)
        yi = xi ** 2

    plt.show()


if __name__ == '__main__':
    plot_on_parable()
