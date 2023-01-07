import jax.numpy as jnp
from .modules import Model
import jax
from functools import partial


class Sequential(Model):
    """A Sequential model is a nueral network with the 1 input that flows down in a sequence through a stack of layers.
       To create this model, just pass a list of layers that you want the inputs to pass through."""

    def __init__(self, layers, dynamic = False):
        super().__init__(layers = layers, dynamic = dynamic)
    
    
    def __call__(self, x):
        last = self.layers[-1].unit()
        x = self.layers[0].__call__(x)
        for layer in self.layers[1:]:
            x = layer.__call__(x[last])
        return jnp.array(x[last])

    def fit(self, x_train, y_train, epochs = 1):
        super().fit(Model = Sequential, x_train=x_train, y_train=y_train, epochs=epochs)