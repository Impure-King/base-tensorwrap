import jax.numpy as jnp
from .modules import Model
import jax
from functools import partial

@jax.tree_util.register_pytree_node_class # Allowing jit compile
class Sequential(Model):
    """A Sequential model is a nueral network with the 1 input that flows down in a sequence through a stack of layers.
       To create this model, just pass a list of layers that you want the inputs to pass through."""

    def __init__(self, layers, dynamic = False):
        super().__init__(layers = layers, dynamic = dynamic)
    
    
    @jax.jit
    def __call__(self, x):
        if self.dynamic == True:
            with jax.disable_jit():
                for layer in self.layers:
                    x = layer.__call__(x)
                return x
        elif not self.dynamic:    
                for layer in self.layers:
                    x = layer.__call__(x)
                return jnp.array(x)
        else:
            raise ValueError("Dynamic is incorrectly specified. Acceptable terms are None, True, or False.")
    
    def fit(self, x_train, y_train, epochs = 1):
        super().fit(Model = Sequential, x_train=x_train, y_train=y_train, epochs=epochs)