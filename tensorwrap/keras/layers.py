from tensorwrap import Module


class Layer(Module):
    """A base layer class that is used to create new JIT enabled layers.
       Acts as the subclass for all layers."""

    def __init__(self, random_seed=1):
        pass

    def add_weights(self, activations):

    def __call__(self):
        pass
