""" TensorWrap's evergrowing prebuilt optimizers. """

from .models.modules import Module
import jax
from functools import partial


class Optimizer(Module):
    def apply_gradients(self, weights):
        kernel = weights[0]
        kernel += self.grads * self.eta
        bias = weights[1]
        bias += self.grads * self.eta
        return kernel, bias

class SGD(Optimizer):

    def __init__(self, learning_rate = 0.01, jit_compile = True):
        super().__init__(
            learning_rate = learning_rate,
            jit_compile = jit_compile
        )
    
    def compute_grads(self, loss_fn, y_true, y_pred):
        self.eta = self.learning_rate
        self.grads = loss_fn(y_true, y_pred)