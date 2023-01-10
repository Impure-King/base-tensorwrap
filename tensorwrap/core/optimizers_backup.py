""" TensorWrap's evergrowing prebuilt optimizers. """

from tensorwrap.keras.models.modules import Module
import jax


class Optimizer(Module):
    def apply_gradients(self, weights):
        weights -= self.grads * self.eta
        return weights

class SGD(Optimizer):

    def __init__(self, learning_rate = 0.01, jit_compile = True):
        super().__init__(
            learning_rate = learning_rate,
            jit_compile = jit_compile
        )
    
    def compute_grads(self, loss_fn, y_true, y_pred, trainable_variables):
        self.eta = self.learning_rate
        grads = loss_fn(y_true, y_pred)
        self.grads = grads()
        