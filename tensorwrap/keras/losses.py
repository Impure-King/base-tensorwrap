from tensorwrap.module import Module


class Loss(Module):
    """A base loss class that is used to create new loss functions.
       Acts as a subclass for all losses, to ensure that it is compatible with PyTrees and XLA."""
    def __init__(self):
        pass

    def __call__(self):
        pass

    def call(self):
        pass
