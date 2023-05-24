import tensorwrap as tf
import jax
from tensorwrap.module import Module
from functools import partial
from jaxtyping import Array

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


    def call(self, weights: Array, grad: Array):
        return weights - self.lr * grad 

    def apply_gradients(self, weights: dict, gradients: dict):
        weights = jax.tree_map(self.call, weights, gradients)
        return weights
