class GradientDescent:
    learning_rate = 0.001

    @staticmethod
    def step(x, dx):
        return x - GradientDescent.learning_rate * dx