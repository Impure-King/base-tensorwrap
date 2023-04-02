import tensorwrap as tf
import jax
from tensorwrap.module import Module


class Optimizer(Module):

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr


class gradient_descent(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(lr=learning_rate)

    def apply_gradients(self, gradients, layers: dict):
        for layer in layers.values():
            kernel = layer.trainable_variables['w']
            bias = layer.trainable_variables['b']
            layer.trainable_variables['w'] = jax.tree_map(lambda x: x + tf.mean(gradients * self.lr), kernel)
            layer.trainable_variables['b'] = jax.tree_map(lambda x: x + tf.mean(gradients * self.lr), bias)
        return layers
