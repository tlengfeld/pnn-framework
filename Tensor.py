class Tensor:
    def __init__(self, elements=None, deltas=None):
        self.elements = elements
        self.deltas = deltas

    def get_shape(self):
        assert(self.elements is not None)
        return self.elements.shape
