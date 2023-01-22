import tensorwrap as tf
import jax
from tensorwrap.module import Module


class Optimizer(Module):

    def __init__(self, lr=0.01):
        self.lr = lr


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(lr=learning_rate)

    def apply_gradients(self, gradients, kernel, bias):
        kernel += gradients * self.lr
        bias += gradients * self.lr
        return [kernel, bias]
