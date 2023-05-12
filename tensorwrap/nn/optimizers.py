import tensorwrap as tf
import jax
from tensorwrap.module import Module
from functools import partial

class Optimizer(Module):

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        if not NotImplemented:
            raise NotImplementedError
    

# Change the naming to conventions:
class gradient_descent(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(lr=learning_rate)

    
    def call(self, weights, grad):
        return weights - self.lr * grad 

    def apply_gradients(self, gradients: dict, weights: dict):
        weights = jax.tree_map(self.call, weights, gradients)
        return weights
